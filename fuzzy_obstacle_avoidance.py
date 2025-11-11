#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CE801 – Task 3 (FINAL)
Fuzzy Obstacle Avoidance with anti-oscillation
- 5 sektör (L, FL, F, FR, R)
- Fuzzy kural tabanı + centroid
- "Tek engel tam karşıda" titremesini önlemek için:
  * Turn-commit (min süre yön kilidi)
  * FL/FR mean tabanlı gap seçimi
  * ω, v IIR low-pass + slew-rate limit
  * deadband + yön belleği (last_turn_dir)
"""

import math
from typing import Optional, List, Tuple
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from utils_fuzzy import tri, trap, centroid

TOPIC_LASER = '/scan'
TOPIC_CMD   = '/cmd_vel'

# --- Genel sınırlar ---
VALID_MIN = 0.03
VALID_MAX = 5.00

# Sektörler (deg)
SECT_LEFT_DEG        = (100.0, 140.0)
SECT_FRONT_LEFT_DEG  = (40.0,   80.0)
SECT_FRONT_DEG       = (-50.0,  +50.0)
SECT_FRONT_RIGHT_DEG = (-80.0,  -40.0)
SECT_RIGHT_DEG       = (-140.0, -100.0)

# Hız limitleri
V_MAX = 0.20
V_MIN = 0.00
ANGULAR_LIM = 0.90

# Engele yaklaşım eşikleri
OA_NEAR    = 0.40
OA_CAUTION = 0.60

# Süzme
FRONT_SMOOTH_A = 0.18   # ön koni min için IIR oranı

# Deadband / histerezis
BIAS_DEADBAND = 0.06
HYST_CLEAR    = 0.05

# Anti-oscillation: turn-commit (s)
COMMIT_T_NEAR    = 0.90
COMMIT_T_CAUTION = 0.60
COMMIT_W_NEAR    = 0.95 * ANGULAR_LIM
COMMIT_W_CAUTION = 0.60 * ANGULAR_LIM
COMMIT_V_NEAR    = 0.06
COMMIT_V_CAUTION = 0.10

# ω, v düşük geçiren filtre
OMEGA_LPF = 0.35
VEL_LPF   = 0.35

# Slew rate limit (per second)
SLEW_W = 2.0   # rad/s^2
SLEW_V = 0.6   # m/s^2

PRINT_EVERY = 15

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class FuzzyOA(Node):
    def __init__(self):
        super().__init__('fuzzy_obstacle_avoidance')
        self.sub = self.create_subscription(LaserScan, TOPIC_LASER, self.on_scan, 10)
        self.pub = self.create_publisher(Twist, TOPIC_CMD, 10)

        self.front_min_filt: Optional[float] = None
        self.last_turn_dir = 1.0  # +left / -right
        self.scan_count = 0

        # commit latch
        self.commit_until: float = 0.0
        self.commit_dir: float = 0.0
        self.commit_omega: float = 0.0
        self.commit_v: float = 0.0

        # filtre ve slew için geçmiş
        self.prev_time = self.now_s()
        self.prev_omega = 0.0
        self.prev_v = 0.0

    # ---------- zaman ----------
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------- yardımcılar ----------
    def _valid(self, v: float) -> bool:
        return (v is not None) and math.isfinite(v) and (VALID_MIN <= v <= VALID_MAX)

    def _idx(self, amin, ainc, n, d0, d1):
        i0 = int(round((math.radians(d0) - amin) / ainc))
        i1 = int(round((math.radians(d1) - amin) / ainc))
        i0 = max(0, min(n, i0)); i1 = max(0, min(n, i1))
        return i0, i1

    def _sector_vals(self, scan: LaserScan, deg0: float, deg1: float) -> List[float]:
        ranges = list(scan.ranges); n = len(ranges)
        i0, i1 = self._idx(scan.angle_min, scan.angle_increment, n, deg0, deg1)
        return [v for v in ranges[i0:i1] if self._valid(v)]

    def _sector_min(self, scan: LaserScan, deg0: float, deg1: float) -> float:
        vals = self._sector_vals(scan, deg0, deg1)
        return min(vals) if vals else math.inf

    def _sector_mean(self, scan: LaserScan, deg0: float, deg1: float) -> float:
        vals = self._sector_vals(scan, deg0, deg1)
        if not vals:
            return math.inf
        return sum(vals) / len(vals)

    def _front_min(self, scan: LaserScan) -> float:
        vals = self._sector_vals(scan, *SECT_FRONT_DEG)
        raw = min(vals) if vals else math.inf
        if self.front_min_filt is None or not math.isfinite(self.front_min_filt):
            self.front_min_filt = raw
        else:
            self.front_min_filt = (1.0 - FRONT_SMOOTH_A) * self.front_min_filt + FRONT_SMOOTH_A * raw
        return self.front_min_filt

    # ---------- fuzzy kümeler ----------
    def fuzz_dist(self, d: float):
        d = max(0.0, min(2.0, d if math.isfinite(d) else 2.0))
        return {
            'VeryClose': trap(d, 0.00, 0.00, 0.35, 0.50),
            'Close'    : tri (d, 0.40, 0.65, 0.95),
            'Far'      : trap(d, 0.85, 1.10, 2.00, 2.00),
        }

    def fuzz_bias(self, b: float):
        b = max(-0.8, min(0.8, b))
        return {
            'RightClear': trap(b, -0.8, -0.8, -0.25, -0.06),
            'Center'    : tri (b, -0.10, 0.0, +0.10),
            'LeftClear' : trap(b, +0.06, +0.25, +0.8, +0.8),
        }

    def ω_sets(self):
        return {
            'HARD_RIGHT': (-ANGULAR_LIM, -0.45, 'Z'),
            'SOFT_RIGHT': (-0.45,        -0.10, 'Z'),
            'STRAIGHT'  : (-0.04,        +0.04, 'tri'),
            'SOFT_LEFT' : (+0.10,        +0.45, 'S'),
            'HARD_LEFT' : (+0.45,  +ANGULAR_LIM, 'S'),
        }

    def v_sets(self):
        return {
            'STOP': (0.00, 0.04, 'tri'),
            'SLOW': (0.06, 0.12, 'S'),
            'FAST': (0.14, V_MAX, 'S'),
        }

    # ---------- kural tabanı ----------
    def rulebase(self, D: dict, bias: dict):
        mu = []; Ω = self.ω_sets(); V = self.v_sets()
        def addΩ(m, s): mu.append((m, Ω[s])) if m > 0.0 else None
        def addV(m, s): mu.append((m, V[s])) if m > 0.0 else None

        # Ön çok yakın → dur + açık tarafa sert dön
        addV(D['F']['VeryClose'], 'STOP')
        addΩ(min(D['F']['VeryClose'], bias['LeftClear']),  'HARD_LEFT')
        addΩ(min(D['F']['VeryClose'], bias['RightClear']), 'HARD_RIGHT')

        # Ön yakın → yavaş + açık tarafa yumuşak dön
        addV(D['F']['Close'], 'SLOW')
        addΩ(min(D['F']['Close'], bias['LeftClear']),  'SOFT_LEFT')
        addΩ(min(D['F']['Close'], bias['RightClear']), 'SOFT_RIGHT')

        # Yan çok yakın → sert uzaklaş
        addΩ(min(D['FL']['VeryClose'], 1.0), 'HARD_RIGHT')
        addΩ(min(D['FR']['VeryClose'], 1.0), 'HARD_LEFT')
        addV(max(D['FL']['VeryClose'], D['FR']['VeryClose']), 'SLOW')

        # Yan yakın → yumuşak uzaklaş
        addΩ(min(D['FL']['Close'], 1.0), 'SOFT_RIGHT')
        addΩ(min(D['FR']['Close'], 1.0), 'SOFT_LEFT')

        # Geniş açıklık → o yöne hafif yönel
        addΩ(min(D['L']['Far'],  bias['LeftClear']),  'SOFT_LEFT')
        addΩ(min(D['R']['Far'],  bias['RightClear']), 'SOFT_RIGHT')

        # Hız modülasyonu
        fast_gate = min(D['F']['Far'], max(D['FL']['Far'], D['FR']['Far']))
        addV(fast_gate, 'FAST')
        danger = max(
            D['F']['Close'], D['F']['VeryClose'],
            D['FL']['Close'], D['FL']['VeryClose'],
            D['FR']['Close'], D['FR']['VeryClose'],
            D['L']['Close'],  D['L']['VeryClose'],
            D['R']['Close'],  D['R']['VeryClose']
        )
        addV(danger, 'SLOW')

        # Düz git
        addΩ(min(D['F']['Far'], bias['Center']), 'STRAIGHT')
        return mu

    # ---------- ana callback ----------
    def on_scan(self, scan: LaserScan):
        t = self.now_s()
        dt = max(1e-3, t - self.prev_time)

        # Sektör istatistikleri
        L_min  = self._sector_min(scan, *SECT_LEFT_DEG)
        FL_min = self._sector_min(scan, *SECT_FRONT_LEFT_DEG)
        F_min  = self._front_min(scan)
        FR_min = self._sector_min(scan, *SECT_FRONT_RIGHT_DEG)
        R_min  = self._sector_min(scan, *SECT_RIGHT_DEG)

        # Gap seçimi için mean (ön yanlar)
        FL_mean = self._sector_mean(scan, *SECT_FRONT_LEFT_DEG)
        FR_mean = self._sector_mean(scan, *SECT_FRONT_RIGHT_DEG)

        # L-R açıklık biası (deadband + bellek)
        raw_bias = 0.0
        if math.isfinite(L_min) and math.isfinite(R_min):
            diff = (L_min - R_min)
            raw_bias = math.tanh(diff)
        if abs(raw_bias) < BIAS_DEADBAND:
            b_eff = self.last_turn_dir * 0.08
        else:
            b_eff = raw_bias
            self.last_turn_dir = 1.0 if b_eff >= 0.0 else -1.0

        # Fuzzy üyelikleri
        D = {
            'L' : self.fuzz_dist(L_min),
            'FL': self.fuzz_dist(FL_min),
            'F' : self.fuzz_dist(F_min),
            'FR': self.fuzz_dist(FR_min),
            'R' : self.fuzz_dist(R_min),
        }
        B = self.fuzz_bias(b_eff)

        # Fuzzy çıktı
        defs = self.rulebase(D, B)
        defs_w = [d for d in defs if d[1][0] < -0.08 or d[1][1] > 0.08]
        defs_v = [d for d in defs if 0.0 <= d[1][0] <= V_MAX and 0.0 <= d[1][1] <= V_MAX]
        omega_fuzzy = clamp(centroid(defs_w) * 1.05, -ANGULAR_LIM, ANGULAR_LIM)
        v_fuzzy     = clamp(centroid(defs_v), V_MIN, V_MAX)

        # --------- Turn-commit mantığı (anti-oscillation) ---------
        # Gap yönü: FL/FR mean kıyasla; eşitlikte bellek
        gap_dir = 0.0
        if math.isfinite(FL_mean) and math.isfinite(FR_mean):
            if abs(FL_mean - FR_mean) > 0.03:
                gap_dir = +1.0 if FL_mean > FR_mean else -1.0
            else:
                gap_dir = self.last_turn_dir
        else:
            gap_dir = self.last_turn_dir

        # Trigger ve latch
        if math.isfinite(F_min):
            if F_min < (OA_NEAR - HYST_CLEAR):
                self._start_commit(t, gap_dir, COMMIT_T_NEAR, COMMIT_W_NEAR, COMMIT_V_NEAR)
            elif F_min < (OA_CAUTION - HYST_CLEAR):
                self._start_commit(t, gap_dir, COMMIT_T_CAUTION, COMMIT_W_CAUTION, COMMIT_V_CAUTION)

        # Commit aktifse override
        omega_des = omega_fuzzy
        v_des     = v_fuzzy
        if t < self.commit_until:
            omega_des = self.commit_dir * self.commit_omega
            v_des     = self.commit_v

        # --------- LPF + Slew limit ---------
        omega_f = self._lpf(self.prev_omega, omega_des, OMEGA_LPF)
        v_f     = self._lpf(self.prev_v,     v_des,     VEL_LPF)

        omega_cmd = self._slew(self.prev_omega, omega_f, SLEW_W * dt)
        v_cmd     = self._slew(self.prev_v,     v_f,     SLEW_V * dt)

        # Yayınla
        tw = Twist()
        tw.linear.x  = clamp(v_cmd, V_MIN, V_MAX)
        tw.angular.z = clamp(omega_cmd, -ANGULAR_LIM, ANGULAR_LIM)
        self.pub.publish(tw)

        # state update
        self.prev_time  = t
        self.prev_omega = tw.angular.z
        self.prev_v     = tw.linear.x

        self.scan_count += 1
        if self.scan_count % PRINT_EVERY == 0:
            self.get_logger().info(
                f"[FuzzyOA] F(min)={F_min:.2f} FLμ={FL_mean:.2f} FRμ={FR_mean:.2f} "
                f"bias={b_eff:+.2f} | v={tw.linear.x:.2f} ω={tw.angular.z:.2f} "
                f"{'| COMMIT' if t<self.commit_until else ''}"
            )

    # --- helpers: commit / filters / slew ---
    def _start_commit(self, now: float, dir_sign: float, dur: float, w: float, v: float):
        # mevcut commit'i güçsüzse uzat
        if now < self.commit_until:
            # sadece daha sert bir moda geçiliyorsa güncelle
            self.commit_until = max(self.commit_until, now + 0.25)
            return
        self.commit_until = now + dur
        self.commit_dir   = 1.0 if dir_sign >= 0.0 else -1.0
        self.last_turn_dir = self.commit_dir
        self.commit_omega = w
        self.commit_v     = v

    def _lpf(self, prev: float, target: float, alpha: float) -> float:
        return (1.0 - alpha) * prev + alpha * target

    def _slew(self, prev: float, target: float, max_delta: float) -> float:
        delta = target - prev
        if delta >  max_delta: delta =  max_delta
        if delta < -max_delta: delta = -max_delta
        return prev + delta

def main():
    rclpy.init()
    node = FuzzyOA()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
