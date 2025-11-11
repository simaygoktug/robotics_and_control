#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, time
from typing import Optional, List, Tuple
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from utils_fuzzy import tri, trap, centroid

TOPIC_LASER='/scan'; TOPIC_CMD='/cmd_vel'

# --- Tunings ---
DESIRED_DISTANCE = 0.50
LINEAR_BASE      = 0.16
ANGULAR_LIM      = 0.90
VALID_MIN=0.03; VALID_MAX=5.0

# Sektör açıları (deg)
SIDE_DEG=30.0
FRONT_OF_SIDE_DEG=25.0
FRONT_DEG=55.0
SIDE_CENTER=-90.0
FOS_CENTER=-67.0

# Süzme
SIDE_SMOOTH_A     = 0.22
FRONT_SMOOTH_A    = 0.28
ANGLE_SMOOTH_A    = 0.25   # yeni: duvar açısı filtresi
OMEGA_LPF_A       = 0.35   # çıkış yumuşatma
VEL_LPF_A         = 0.25
SLEW_W            = 2.0    # rad/s^2
SLEW_V            = 0.6    # m/s^2

# Acil-durum eşikleri
OA_NEAR    = 0.40
OA_CAUTION = 0.60

def clamp(x, lo, hi): return max(lo, min(hi, x))

class FuzzyRightEdgeParallel(Node):
    def __init__(self):
        super().__init__('fuzzy_right_edge_parallel')
        self.sub=self.create_subscription(LaserScan, TOPIC_LASER, self.on_scan, 10)
        self.pub=self.create_publisher(Twist, TOPIC_CMD, 10)

        self.filtered_side=None
        self.front_min_filt=None
        self.theta_filt=None         # duvar açısı filtresi (rad)
        self.last_e=None; self.last_t=None
        self.last_turn_dir=1.0
        self.scan_count=0

        # çıkış filtre/slew hafızası
        self.prev_time=time.monotonic()
        self.prev_omega=0.0
        self.prev_v=0.0

    # ---------- yardımcılar ----------
    def _valid(self, v):
        return (v is not None) and math.isfinite(v) and VALID_MIN <= v <= VALID_MAX
    def _nanmean(self, xs):
        xs=[x for x in xs if x is not None and math.isfinite(x) and VALID_MIN<=x<=VALID_MAX]
        return sum(xs)/len(xs) if xs else math.nan
    def _idx(self, amin, ainc, n, d0, d1):
        i0=int(round((math.radians(d0)-amin)/ainc)); i1=int(round((math.radians(d1)-amin)/ainc))
        i0=max(0,min(n,i0)); i1=max(0,min(n,i1)); return i0,i1

    def estimate_side_and_angle(self, scan):
        """
        Sağ duvar ölçümü + duvar açısı (phi) tahmini.
        phi > 0: ileri baktıkça duvar uzaklaşıyor (robot duvardan YÜZEYSEL olarak uzak bakıyor) -> sağa dönmeli
        phi < 0: ileri baktıkça duvar yaklaşıyor (robot duvara doğru bakıyor) -> sola dönmeli
        """
        ranges=list(scan.ranges); n=len(ranges)
        s0,s1=self._idx(scan.angle_min, scan.angle_increment, n, SIDE_CENTER-SIDE_DEG/2, SIDE_CENTER+SIDE_DEG/2)
        f0,f1=self._idx(scan.angle_min, scan.angle_increment, n, FOS_CENTER-FRONT_OF_SIDE_DEG/2, FOS_CENTER+FRONT_OF_SIDE_DEG/2)
        d_side=self._nanmean(ranges[s0:s1]); d_fos=self._nanmean(ranges[f0:f1])

        phi=None
        if self._valid(d_side) and self._valid(d_fos):
            theta_sep=math.radians(abs(FOS_CENTER-SIDE_CENTER))
            if theta_sep>1e-3:
                # duvar normaline göre yaklaşık açı
                phi = math.atan2(d_fos - d_side, d_side*math.tan(theta_sep))  # rad

        # perpendicular düzeltmeli mesafe (opsiyonel)
        result = d_side
        if self._valid(d_side) and (phi is not None):
            d_perp = d_side*math.cos(phi)
            if self._valid(d_perp) and abs(d_perp-d_side) < 0.25:
                result = d_perp

        # filtreler
        if self.filtered_side is None and self._valid(result):
            self.filtered_side = result
        elif self._valid(result):
            self.filtered_side=(1.0-SIDE_SMOOTH_A)*self.filtered_side + SIDE_SMOOTH_A*result

        if phi is not None:
            if self.theta_filt is None:
                self.theta_filt = phi
            else:
                self.theta_filt = (1.0-ANGLE_SMOOTH_A)*self.theta_filt + ANGLE_SMOOTH_A*phi

        return self.filtered_side, self.theta_filt

    def front_min(self, scan):
        n=len(scan.ranges)
        i0,i1=self._idx(scan.angle_min, scan.angle_increment, n,-FRONT_DEG/2,FRONT_DEG/2)
        vals=[v for v in scan.ranges[i0:i1] if self._valid(v)]
        raw=min(vals) if vals else math.inf
        if self.front_min_filt is None or not math.isfinite(self.front_min_filt):
            self.front_min_filt=raw
        else:
            self.front_min_filt=(1.0-FRONT_SMOOTH_A)*self.front_min_filt + FRONT_SMOOTH_A*raw
        return self.front_min_filt

    def side_stats(self, ranges, scan):
        n=len(ranges)
        L0,L1=self._idx(scan.angle_min, scan.angle_increment, n, 60.0,120.0)
        R0,R1=self._idx(scan.angle_min, scan.angle_increment, n,-120.0,-60.0)
        L=[v for v in ranges[L0:L1] if self._valid(v)]
        R=[v for v in ranges[R0:R1] if self._valid(v)]
        lmin=min(L) if L else math.inf; rmin=min(R) if R else math.inf
        lmean=(sum(L)/len(L)) if L else math.inf; rmean=(sum(R)/len(R)) if R else math.inf
        return lmin, rmin, lmean, rmean

    # ---------- fuzzy kümeler ----------
    def fuzz_error(self, e):
        e=max(-0.6,min(0.6,e))
        return {
            'NL': trap(e,-0.6,-0.6,-0.40,-0.22),
            'NS': tri (e,-0.32,-0.18,-0.04),
            'ZE': tri (e,-0.08, 0.00, 0.08),
            'PS': tri (e, 0.04, 0.18, 0.32),
            'PL': trap(e, 0.22, 0.40, 0.60, 0.60),
        }

    def fuzz_theta(self, th):
        # th (rad): + → duvardan uzak bakıyor; - → duvara bakıyor
        th=max(-0.45, min(0.45, th if th is not None else 0.0))
        # yaklaşık ±25° aralık
        return {
            'Toward'  : trap(th, -0.45, -0.45, -0.12, -0.03),  # duvara dönük
            'Parallel': tri (th, -0.05,  0.00,  +0.05),
            'Away'    : trap(th, +0.03,  +0.12, +0.45, +0.45), # duvardan uzak bakıyor
        }

    def fuzz_front(self, df):
        df=max(0.0,min(2.0,df))
        return {
            'VeryClose': trap(df,0.00,0.00,0.35,0.50),
            'Close'    : tri (df,0.45,0.65,0.95),
            'Far'      : trap(df,0.85,1.10,2.0,2.0),
        }

    def ω_sets(self):
        return {
            'HARD_RIGHT': (-ANGULAR_LIM,-0.45,'Z'),
            'SOFT_RIGHT': (-0.45,-0.10,'Z'),
            'STRAIGHT'  : (-0.04, 0.04,'tri'),
            'SOFT_LEFT' : ( 0.10, 0.45,'S'),
            'HARD_LEFT' : ( 0.45, ANGULAR_LIM,'S'),
        }

    def v_sets(self):
        return {
            'STOP': (0.00,0.05,'tri'),
            'SLOW': (0.06,0.12,'S'),
            'FAST': (0.12,0.20,'S'),
        }

    def rulebase(self, E, TH, DF):
        mu=[]; Ω=self.ω_sets(); V=self.v_sets()
        def addΩ(m, s): mu.append((m, Ω[s])) if m>0 else None
        def addV(m, s): mu.append((m, V[s])) if m>0 else None

        # --- Obstacle önceliği ---
        addV(DF['VeryClose'], 'STOP')
        addΩ(DF['VeryClose'], 'HARD_LEFT')   # sağ duvar takibinde öne engelde sola kaç
        addV(DF['Close'], 'SLOW')
        addΩ(DF['Close'], 'SOFT_LEFT')

        # --- Paralel tutma (yeni: açıya dayalı) (DF Far iken baskın) ---
        addΩ(min(DF['Far'], TH['Away']),    'SOFT_RIGHT')  # uzak bakıyorsan sağa dön
        addΩ(min(DF['Far'], TH['Toward']),  'SOFT_LEFT')   # duvara bakıyorsan sola dön
        addΩ(min(DF['Far'], TH['Parallel']), 'STRAIGHT')

        # --- Mesafe düzeltmesi (DF Far) ---
        addΩ(min(DF['Far'], E['PL']), 'SOFT_LEFT')   # duvara yakınsan uzaklaş (sola dön)
        addΩ(min(DF['Far'], E['PS']), 'SOFT_LEFT')
        addΩ(min(DF['Far'], E['NL']), 'SOFT_RIGHT')  # duvardan uzaklaştıysan yaklaş (sağa dön)
        addΩ(min(DF['Far'], E['NS']), 'SOFT_RIGHT')

        # --- Hız modülasyonu ---
        addV(min(DF['Far'], E['ZE'], TH['Parallel']), 'FAST')
        addV(max(E['PL'],E['NL'], TH['Away'], TH['Toward']), 'SLOW')

        return mu

    # ---------- ana callback ----------
    def on_scan(self, scan: LaserScan):
        ranges=list(scan.ranges)
        d_side, theta = self.estimate_side_and_angle(scan)
        d_front=self.front_min(scan)

        # acil-durum (OA)
        if math.isfinite(d_front):
            tw = Twist()
            if d_front < OA_NEAR:
                tw.linear.x = 0.0
                tw.angular.z = +ANGULAR_LIM   # sola sert (sağ duvar takipte)
                self.pub.publish(tw); return
            elif d_front < OA_CAUTION:
                tw.linear.x = 0.08
                tw.angular.z = +0.6*ANGULAR_LIM
                self.pub.publish(tw); return

        if d_side is None or not self._valid(d_side):
            tw=Twist(); tw.linear.x=LINEAR_BASE*0.7; tw.angular.z=0.0
            self.pub.publish(tw); return

        # hata & açı
        e=DESIRED_DISTANCE - d_side
        tnow=time.monotonic()
        if self.last_t is None:
            ed=0.0
        else:
            dt=max(1e-3, tnow - self.last_t)
            ed=(e - (self.last_e or 0.0))/dt
            ed=max(-1.0,min(1.0,ed))
        self.last_e,self.last_t=e,tnow

        # fuzzy
        E=self.fuzz_error(e)
        TH=self.fuzz_theta(theta if theta is not None else 0.0)
        DF=self.fuzz_front(d_front if math.isfinite(d_front) else 2.0)
        defs=self.rulebase(E, TH, DF)

        defs_w=[d for d in defs if abs(d[1][0])>0.06 or abs(d[1][1])>0.06]
        defs_v=[d for d in defs if 0.0<=d[1][0]<=0.3 and 0.0<=d[1][1]<=0.3]
        omega=centroid(defs_w)
        v=centroid(defs_v)

        # küçük feedforward: açı hatasına oransal term (paralellik ısrarı)
        if theta is not None:
            omega += clamp(-0.6*theta, -0.25, 0.25)  # θ>0 → sağa küçük itki

        # filtre + slew
        now=time.monotonic()
        dt=max(1e-3, now - self.prev_time)
        omega = (1.0-OMEGA_LPF_A)*self.prev_omega + OMEGA_LPF_A*omega
        v     = (1.0-VEL_LPF_A)*self.prev_v     + VEL_LPF_A*v

        max_dw=SLEW_W*dt; max_dv=SLEW_V*dt
        domega = clamp(omega - self.prev_omega, -max_dw, max_dw)
        dv     = clamp(v - self.prev_v, -max_dv, max_dv)
        omega_cmd = clamp(self.prev_omega + domega, -ANGULAR_LIM, ANGULAR_LIM)
        v_cmd     = clamp(self.prev_v + dv, 0.0, 0.20)

        tw=Twist()
        tw.linear.x  = v_cmd if DF['VeryClose']<0.5 else max(v_cmd, 0.10)  # çok yakın değilsek tabanı koru
        tw.angular.z = omega_cmd
        self.pub.publish(tw)

        self.prev_time=now; self.prev_omega=omega_cmd; self.prev_v=v_cmd

def main():
    rclpy.init()
    node=FuzzyRightEdgeParallel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
