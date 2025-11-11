#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
import os, time, math

FRONT_DEG = 30.0         # ±15°
RIGHT_CTR = -90.0        # sağ yan merkezi
RIGHT_WID = 20.0         # sağ pencere genişliği

class PerformanceObserver(Node):
    def __init__(self):
        super().__init__('performance_observer')
        self.sub_cmd  = self.create_subscription(Twist,     '/cmd_vel', self.cmd_cb,   50)
        self.sub_scan = self.create_subscription(LaserScan, '/scan',    self.scan_cb,  50)
        self.sub_odom = self.create_subscription(Odometry,  '/odom',    self.odom_cb,  50)

        self.t0 = time.time()
        self.t_cmd, self.v, self.w = [], [], []          # cmd zaman serileri
        self.te, self.err, self.front_d, self.right_d = [], [], [], []  # error zaman serileri
        self.x, self.y = [], []

        self.last_scan = None
        self.last_scan_meta = None
        self.target_dist = 0.40

        self.outdir = os.path.expanduser('~/M-Drive/CE801/performance_results')
        os.makedirs(self.outdir, exist_ok=True)
        self.get_logger().info("📡 Observer (read-only) listening: /cmd_vel, /scan, /odom")

    # ---------- helpers ----------
    def _now(self): return time.time() - self.t0

    def _idx_range(self, scan: LaserScan, deg0, deg1):
        """return slice indices for [deg0, deg1] (deg) based on scan angular meta"""
        n = len(scan.ranges)
        a0 = scan.angle_min
        inc = scan.angle_increment
        def idx_for_deg(d):
            r = math.radians(d)
            i = int(round((r - a0) / inc))
            return max(0, min(n-1, i))
        i0, i1 = idx_for_deg(deg0), idx_for_deg(deg1)
        if i0 <= i1: 
            return slice(i0, i1+1)
        # wrap-around
        return list(range(i0, n)) + list(range(0, i1+1))

    def _sector_mean(self, scan: LaserScan, deg_center, deg_width):
        d0 = deg_center - deg_width/2.0
        d1 = deg_center + deg_width/2.0
        sel = self._idx_range(scan, d0, d1)
        if isinstance(sel, slice):
            vals = [v for v in scan.ranges[sel] if self._valid(v)]
        else:
            vals = [scan.ranges[i] for i in sel if self._valid(scan.ranges[i])]
        return float(np.mean(vals)) if len(vals) else np.nan

    def _valid(self, v):
        return (v is not None) and math.isfinite(v) and 0.03 <= v <= 5.0

    # ---------- callbacks ----------
    def cmd_cb(self, msg: Twist):
        t = self._now()
        self.t_cmd.append(t)
        self.v.append(msg.linear.x)
        self.w.append(msg.angular.z)
        # not: hata burada eklenmiyor; scan gelince ekliyoruz (kendi zamanıyla)

    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg.ranges
        self.last_scan_meta = msg
        # ölçü anında error hesapla & zamanla birlikte kaydet
        t = self._now()
        f = self._sector_mean(msg, 0.0, FRONT_DEG)                    # ±15°
        r = self._sector_mean(msg, RIGHT_CTR, RIGHT_WID)              # sağ pencere
        if not math.isnan(r):
            self.te.append(t)
            self.right_d.append(r)
            self.front_d.append(f if not math.isnan(f) else np.nan)
            self.err.append(self.target_dist - r)

    def odom_cb(self, msg: Odometry):
        self.x.append(msg.pose.pose.position.x)
        self.y.append(msg.pose.pose.position.y)

    # ---------- finalize ----------
    def save_results(self):
        if len(self.t_cmd) < 5 or len(self.te) < 5:
            self.get_logger().warn("Not enough samples to analyze.")
            return

        t_cmd = np.array(self.t_cmd)
        v     = np.array(self.v)
        w     = np.array(self.w)

        te    = np.array(self.te)
        e     = np.array(self.err)
        rd    = np.array(self.right_d)

        # mask: yalnızca error örnekleri
        m = np.isfinite(e)
        te, e, rd = te[m], e[m], rd[m]
        if len(te) < 5:
            self.get_logger().warn("Too few valid error samples after filtering.")
            return

        # ∫|e|dt için te ekseninde trapz
        iae = np.trapz(np.abs(e), te)

        # yerleşme: |e|<2 cm koşulu sağlandıktan sonraki son zamana bak
        settled_idx = np.where(np.abs(e) < 0.02)[0]
        settling = te[settled_idx[-1]] if len(settled_idx) else np.nan

        # overshoot: sağ mesafede hedefin üstüne çıkma yüzdesi (pozitif yönde)
        overshoot = 0.0
        if np.isfinite(rd).any():
            overshoot = max(0.0, (np.nanmax(rd) - self.target_dist) / max(1e-6, self.target_dist) * 100.0)

        self.get_logger().info(f"📈 IAE={iae:.4f}, Overshoot={overshoot:.2f}%, Settling≈{settling:.2f}s")
        self._plot_and_save(t_cmd, v, w, te, e, rd)

    def _plot_and_save(self, t_cmd, v, w, te, e, rd):
        # Error & omega
        plt.figure(figsize=(12,6))
        ax1 = plt.subplot(2,1,1)
        plt.plot(te, e, label='Distance Error (target - right)')
        plt.axhline(0, ls='--', color='k', lw=1)
        plt.ylabel('Error [m]'); plt.grid(True); plt.legend()
        ax2 = plt.subplot(2,1,2, sharex=ax1)
        plt.plot(t_cmd, w, label='Angular Velocity ω')
        plt.xlabel('Time [s]'); plt.ylabel('ω [rad/s]')
        plt.grid(True); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, 'error_response.png'), dpi=200)

        # Right distance trend (opsiyonel yararlı)
        plt.figure(figsize=(10,4))
        plt.plot(te, rd, label='Right distance (scan)')
        plt.axhline(self.target_dist, ls='--', color='k', lw=1, label='Target')
        plt.xlabel('Time [s]'); plt.ylabel('Right distance [m]')
        plt.grid(True); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, 'right_distance.png'), dpi=200)

        # Velocities
        plt.figure(figsize=(10,5))
        plt.plot(t_cmd, v, label='Linear v')
        plt.plot(t_cmd, w, label='Angular ω')
        plt.xlabel('Time [s]'); plt.ylabel('Velocity')
        plt.grid(True); plt.legend(); plt.title('Command Velocities')
        plt.savefig(os.path.join(self.outdir, 'velocities.png'), dpi=200)

        # Trajectory (if any)
        if len(self.x) > 2:
            plt.figure(figsize=(8,6))
            plt.plot(self.x, self.y, 'b-', label='Trajectory (/odom)')
            plt.axis('equal'); plt.grid(True)
            plt.xlabel('x [m]'); plt.ylabel('y [m]')
            plt.title('Robot Path'); plt.legend()
            plt.savefig(os.path.join(self.outdir, 'trajectory.png'), dpi=200)

        self.get_logger().info(f"✅ Saved figures to {self.outdir}")

def main():
    rclpy.init()
    node = PerformanceObserver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.save_results()
        except Exception as ex:
            node.get_logger().error(f"save_results error: {ex}")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
