#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CE801 – Task 2 & 3 (COMBINED FINAL)
Fuzzy Right-Edge Following  +  Fuzzy Obstacle Avoidance
- Tek düğümde iki davranışın bulanık mimarisi ve entegrasyonu
- OA tarafında anti-oscillation: Turn-commit (yön kilidi), IIR LPF, slew-rate limit
- Edge-following tarafında hata (e), türevi (edot) ve açı (theta) ile duvar takibi
- Davranış birleştirme: Ön mesafe üyeliklerine dayalı ağırlıklı harmanlama (blend)
"""

import math
from typing import Optional, List
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

# Süzme katsayıları
FRONT_SMOOTH_A = 0.18
SMOOTH_A_SIDE  = 0.22
SMOOTH_A_THETA = 0.25

# Deadband / histerezis
BIAS_DEADBAND = 0.06

# Düşük geçiren filtre ve slew rate limitleri
OMEGA_LPF = 0.35
VEL_LPF   = 0.35
SLEW_W = 2.0
SLEW_V = 0.6

# --- Right-edge parametreleri ---
DESIRED_DISTANCE = 0.50
SIDE_DEG          = 30.0
FRONT_OF_SIDE_DEG = 25.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class FuzzyIntegratedOAEdge(Node):
    def __init__(self):
        super().__init__('fuzzy_integrated_oa_edge')
        self.sub = self.create_subscription(LaserScan, TOPIC_LASER, self.on_scan, 10)
        self.pub = self.create_publisher(Twist, TOPIC_CMD, 10)

        # filtreler/bellekler
        self.front_min_filt: Optional[float] = None
        self.filtered_side: Optional[float]  = None
        self.theta_filt: Optional[float]     = 0.0
        self.last_turn_dir = 1.0

        # zaman
        self.prev_t  = self.now_s()
        self.prev_e  = 0.0

        # LPF + slew
        self.prev_time  = self.prev_t
        self.prev_omega = 0.0
        self.prev_v     = 0.0


    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


    def _valid(self, v: float) -> bool:
        return (v is not None) and math.isfinite(v) and (VALID_MIN <= v <= VALID_MAX)


    def _idx(self, amin, ainc, n, d0, d1):
        i0 = int(round((math.radians(d0) - amin) / ainc))
        i1 = int(round((math.radians(d1) - amin) / ainc))
        i0 = max(0, min(n, i0)); i1 = max(0, min(n, i1))
        return i0, i1


    def _vals(self, scan: LaserScan, deg0: float, deg1: float) -> List[float]:
        ranges = list(scan.ranges)
        n = len(ranges)
        i0, i1 = self._idx(scan.angle_min, scan.angle_increment, n, deg0, deg1)
        return [v for v in ranges[i0:i1] if self._valid(v)]


    def _sector_min(self, scan: LaserScan, deg0: float, deg1: float) -> float:
        vals = self._vals(scan, deg0, deg1)
        return min(vals) if vals else math.inf


    def _front_min(self, scan: LaserScan) -> float:
        vals = self._vals(scan, *SECT_FRONT_DEG)
        raw = min(vals) if vals else math.inf
        if self.front_min_filt is None or not math.isfinite(self.front_min_filt):
            self.front_min_filt = raw
        else:
            self.front_min_filt = (1.0 - FRONT_SMOOTH_A) * self.front_min_filt + FRONT_SMOOTH_A * raw
        return self.front_min_filt


    # --- sağ duvar mesafe ve açı kestirimi ---
    def estimate_right_side(self, scan: LaserScan):
        side_center = -90.0
        fos_center  = -67.0
        s_vals = self._vals(scan, side_center - SIDE_DEG/2, side_center + SIDE_DEG/2)
        f_vals = self._vals(scan, fos_center  - FRONT_OF_SIDE_DEG/2, fos_center  + FRONT_OF_SIDE_DEG/2)
        d_side = (sum(s_vals)/len(s_vals)) if s_vals else math.nan
        d_fos  = (sum(f_vals)/len(f_vals)) if f_vals else math.nan

        theta = 0.0
        if math.isnan(d_side):
            return math.nan, 0.0
        if not math.isnan(d_fos):
            theta = math.atan2(d_fos - d_side, 0.25)
        return d_side, theta


    # --- fuzzy kümeler ---
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


    def fuzz_error(self, e):
        e = max(-0.6, min(0.6, e))
        return {
            'NL': trap(e,-0.6,-0.6,-0.40,-0.22),
            'NS': tri (e,-0.32,-0.18,-0.04),
            'ZE': tri (e,-0.08, 0.00, 0.08),
            'PS': tri (e, 0.04, 0.18, 0.32),
            'PL': trap(e, 0.22, 0.40, 0.60, 0.60),
        }


    def fuzz_edot(self, ed):
        ed = max(-1.0, min(1.0, ed))
        return {
            'ND': trap(ed,-1.0,-1.0,-0.6,-0.2),
            'ZE': tri (ed,-0.10,0.0,0.10),
            'PD': trap(ed, 0.2, 0.6, 1.0, 1.0),
        }


    def fuzz_theta(self, th):
        th = max(-0.5, min(0.5, th))
        return {
            'Toward'  : trap(th, -0.5, -0.5, -0.15, -0.05),
            'Parallel': tri (th, -0.08, 0.0, 0.08),
            'Away'    : trap(th,  0.05, 0.15, 0.5, 0.5),
        }


    # ---------- OBSTACLE AVOIDANCE ----------
    def rulebase_oa(self, D: dict, bias: dict):
        mu = []
        Ω = self.ω_sets()
        V = self.v_sets()

        def addΩ(m, s): 
            if m > 0: mu.append((m, Ω[s]))
        def addV(m, s): 
            if m > 0: mu.append((m, V[s]))

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

        # Düz git
        addΩ(min(D['F']['Far'], bias['Center']), 'STRAIGHT')
        return mu


    # ---------- EDGE FOLLOWING ----------
    def rulebase_edge(self, E, ED, DF, TH):
        mu=[]; Ω=self.ω_sets(); V=self.v_sets()
        def addΩ(m, s): mu.append((m, Ω[s])) if m>0 else None
        def addV(m, s): mu.append((m, V[s])) if m>0 else None

        addΩ(min(DF['Far'], E['PL']), 'SOFT_LEFT')
        addΩ(min(DF['Far'], E['PS']), 'SOFT_LEFT')
        addΩ(min(DF['Far'], E['NL']), 'SOFT_RIGHT')
        addΩ(min(DF['Far'], E['NS']), 'SOFT_RIGHT')
        addΩ(min(DF['Far'], E['ZE'], ED['ZE']), 'STRAIGHT')

        # Açı bazlı düzeltmeler
        addΩ(min(TH['Toward'], 0.7), 'SOFT_LEFT')
        addΩ(min(TH['Away'],   0.7), 'SOFT_RIGHT')
        addΩ(TH['Parallel'], 'STRAIGHT')

        addV(min(DF['Far'], E['ZE']), 'FAST')
        addV(min(DF['Far'], max(E['PL'],E['NL'])), 'SLOW')
        addV(max(ED['PD'], ED['ND']), 'SLOW')
        addV(DF['Close'], 'SLOW')
        addV(DF['VeryClose'], 'STOP')
        return mu


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


    # --- on_scan ---
    def on_scan(self, scan: LaserScan):
        t  = self.now_s()
        dt = max(1e-3, t - self.prev_time)

        L_min  = self._sector_min(scan, *SECT_LEFT_DEG)
        FL_min = self._sector_min(scan, *SECT_FRONT_LEFT_DEG)
        F_min  = self._front_min(scan)
        FR_min = self._sector_min(scan, *SECT_FRONT_RIGHT_DEG)
        R_min  = self._sector_min(scan, *SECT_RIGHT_DEG)

        raw_bias = 0.0
        if math.isfinite(L_min) and math.isfinite(R_min):
            diff = (L_min - R_min)
            raw_bias = math.tanh(diff)
        if abs(raw_bias) < BIAS_DEADBAND:
            b_eff = self.last_turn_dir * 0.08
        else:
            b_eff = raw_bias
            self.last_turn_dir = 1.0 if b_eff >= 0.0 else -1.0

        D = {'L': self.fuzz_dist(L_min), 'FL': self.fuzz_dist(FL_min),
             'F': self.fuzz_dist(F_min), 'FR': self.fuzz_dist(FR_min), 'R': self.fuzz_dist(R_min)}
        B = self.fuzz_bias(b_eff)

        defs_oa = self.rulebase_oa(D, B)
        defs_w_oa = [d for d in defs_oa if abs(d[1][0]) > 0.08 or abs(d[1][1]) > 0.08]
        defs_v_oa = [d for d in defs_oa if d[1][0] >= 0.0]
        omega_oa = clamp(centroid(defs_w_oa) * 1.05, -ANGULAR_LIM, ANGULAR_LIM)
        v_oa     = clamp(centroid(defs_v_oa), V_MIN, V_MAX)

        d_side, theta = self.estimate_right_side(scan)
        if self.filtered_side is None:
            self.filtered_side = d_side
        elif self._valid(d_side):
            self.filtered_side = (1-SMOOTH_A_SIDE)*self.filtered_side + SMOOTH_A_SIDE*d_side

        if self.theta_filt is None:
            self.theta_filt = theta
        else:
            self.theta_filt = (1-SMOOTH_A_THETA)*self.theta_filt + SMOOTH_A_THETA*theta

        e = DESIRED_DISTANCE - self.filtered_side if self._valid(self.filtered_side) else 0.0
        dt_e = max(1e-3, t - self.prev_t)
        ed = (e - self.prev_e) / dt_e
        self.prev_e = e
        self.prev_t = t

        E  = self.fuzz_error(e)
        ED = self.fuzz_edot(ed)
        DF = self.fuzz_dist(F_min)
        TH = self.fuzz_theta(self.theta_filt)

        defs_edge = self.rulebase_edge(E, ED, DF, TH)
        defs_w_edge = [d for d in defs_edge if abs(d[1][0]) > 0.08]
        defs_v_edge = [d for d in defs_edge if d[1][0] >= 0.0]
        omega_edge = clamp(centroid(defs_w_edge)*1.15, -ANGULAR_LIM, ANGULAR_LIM)
        v_edge     = clamp(centroid(defs_v_edge), V_MIN, V_MAX)

        # açı paralellik düzeltmesi
        omega_edge -= 0.6 * self.theta_filt

        w_oa = max(D['F']['VeryClose'], D['F']['Close'])
        w_edge = 1.0 - w_oa
        omega_des = w_oa*omega_oa + w_edge*omega_edge
        v_des     = w_oa*v_oa     + w_edge*v_edge

        omega_f = self._lpf(self.prev_omega, omega_des, OMEGA_LPF)
        v_f     = self._lpf(self.prev_v, v_des, VEL_LPF)

        domega = clamp(omega_f - self.prev_omega, -SLEW_W*dt, SLEW_W*dt)
        dv     = clamp(v_f - self.prev_v, -SLEW_V*dt, SLEW_V*dt)
        omega_cmd = self.prev_omega + domega
        v_cmd     = self.prev_v + dv

        tw = Twist()
        tw.linear.x  = v_cmd
        tw.angular.z = omega_cmd
        self.pub.publish(tw)

        self.prev_omega = omega_cmd
        self.prev_v = v_cmd
        self.prev_time = t


    def _lpf(self, prev, val, a):
        return (1 - a)*prev + a*val


def main():
    rclpy.init()
    node = FuzzyIntegratedOAEdge()
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
