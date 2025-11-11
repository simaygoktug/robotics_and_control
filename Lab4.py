#!/usr/bin/env python3
"""
CE801 - Lab 4 - FINALIZED VERSION v2
Right-Wall Following + Obstacle Avoidance (Subsumption Architecture)
- Increased target distance to prevent wall collision
- Balanced PID tuning for smooth tracking
- Enhanced distance estimation
"""

from typing import List, Tuple, Optional
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time

TOPIC_LASER = '/scan'
TOPIC_CMD   = '/cmd_vel'

# --- Wall Following Configuration ---
FOLLOW_SIDE = 'RIGHT'
DESIRED_DISTANCE = 0.50     # [m] INCREASED: safer distance from wall
LINEAR_SPEED = 0.16         # [m/s] slightly reduced for better control

# PID gains - Balanced for smooth, stable tracking
KP = 0.7                    # Proportional: reduced to prevent overshooting
KI = 0.015                  # Integral: minimal to handle steady-state
KD = 0.18                   # Derivative: increased for better damping

ANGULAR_LIMIT = 0.9         # [rad/s] slightly reduced max turn rate
I_CLAMP = 0.4               # integral anti-windup

# LiDAR filtering
VALID_MIN = 0.03
VALID_MAX = 5.00
SMOOTHING_ALPHA = 0.22      # Balanced smoothing

# Sensor sector angles
SIDE_DEG = 30.0             # Narrower for precise side reading
FRONT_OF_SIDE_DEG = 25.0    # Front-of-side sector

# --- Obstacle Avoidance Configuration ---
FRONT_DEG = 55.0            # Front detection cone
OA_NEAR = 0.32              # [m] emergency stop distance
OA_CAUTION = 0.50           # [m] slow down distance
OA_EXIT_CLEAR = 0.62        # [m] clearance for exit (hysteresis)

# Avoidance behavior speeds
OA_SLOW = 0.10              # [m/s] speed during caution
OA_SPIN = 0.0               # [m/s] speed during emergency turn
OA_TURN = 0.85              # fraction of ANGULAR_LIMIT

# Anti-jitter parameters
ESCAPE_TIME = 0.6           # [s] commitment time
BIAS_DEADBAND = 0.12        # deadband to prevent oscillation
FRONT_SMOOTH_ALPHA = 0.28   # front distance smoothing

PRINT_EVERY = 10

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value between limits."""
    return max(lo, min(hi, x))

class PID:
    """PID controller with anti-windup."""
    def __init__(self, kp: float, ki: float, kd: float, i_clamp: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_clamp = abs(i_clamp)
        
        self.e_prev = 0.0
        self.i_term = 0.0
        self.t_prev: Optional[float] = None

    def reset(self):
        """Reset controller state."""
        self.e_prev = 0.0
        self.i_term = 0.0
        self.t_prev = None

    def step(self, e: float, now_s: float) -> float:
        """Calculate PID output."""
        if self.t_prev is None:
            dt = 0.02
        else:
            dt = max(1e-3, now_s - self.t_prev)

        # Integral with anti-windup
        self.i_term += e * dt
        self.i_term = clamp(self.i_term, -self.i_clamp, self.i_clamp)

        # Derivative with filtering
        d = (e - self.e_prev) / dt if dt > 0 else 0.0
        
        # PID output
        u = self.kp * e + self.ki * self.i_term + self.kd * d

        self.e_prev = e
        self.t_prev = now_s
        return u

class WallFollower(Node):
    """
    ROS2 Node implementing subsumption architecture:
    Priority 1 (Highest): Obstacle Avoidance
    Priority 2 (Lower): Right-Wall Following
    """
    
    def __init__(self):
        super().__init__('lab4_wall_follower_final')
        self.get_logger().info('Lab4 FINAL v2: Right-Wall Following + Obstacle Avoidance')
        self.get_logger().info(f'Target distance: {DESIRED_DISTANCE}m | Speed: {LINEAR_SPEED}m/s')

        # ROS2 interfaces
        self.sub = self.create_subscription(LaserScan, TOPIC_LASER, self.on_scan, 10)
        self.pub = self.create_publisher(Twist, TOPIC_CMD, 10)

        # State variables
        self.scan_meta = None
        self.filtered_dist: Optional[float] = None
        self.pid = PID(KP, KI, KD, I_CLAMP)
        self.scan_count = 0

        # Obstacle avoidance state
        self.front_min_filt: Optional[float] = None
        self.state = 'NORMAL'
        self.last_turn_dir = 1.0
        self.escape_until: Optional[float] = None
        
        # Distance tracking for debugging
        self.min_dist_seen = float('inf')
        self.max_dist_seen = 0.0

    # ========== LiDAR Processing Methods ==========
    
    @staticmethod
    def _valid(v: float) -> bool:
        """Check if distance reading is valid."""
        if v is None or math.isnan(v) or math.isinf(v):
            return False
        return VALID_MIN <= v <= VALID_MAX

    @staticmethod
    def _subset_ranges(ranges: List[float], start_idx: int, end_idx: int) -> List[float]:
        """Extract valid readings from index range."""
        start_idx = max(0, start_idx)
        end_idx = min(len(ranges), end_idx)
        return [v for v in ranges[start_idx:end_idx] if WallFollower._valid(v)]

    @staticmethod
    def _nanmean(xs: List[float]) -> float:
        """Calculate mean, returns nan if empty."""
        if not xs:
            return math.nan
        return sum(xs) / float(len(xs))

    def _indices_for_degs(self, angle_min: float, angle_inc: float, n: int, 
                          deg_start: float, deg_end: float) -> Tuple[int, int]:
        """Convert degree range to LiDAR indices."""
        rad = math.radians
        i_start = int(round((rad(deg_start) - angle_min) / angle_inc))
        i_end   = int(round((rad(deg_end)   - angle_min) / angle_inc))
        return (int(clamp(i_start, 0, n)), int(clamp(i_end, 0, n)))

    def estimate_side_distance(self, scan: LaserScan) -> float:
        """
        Estimate perpendicular distance to wall.
        Uses main side sensor with optional front correction.
        """
        ranges = list(scan.ranges)
        n = len(ranges)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment

        # Define sensor regions
        if FOLLOW_SIDE.upper() == 'RIGHT':
            side_center = -90.0
            front_of_side_center = -67.0
        else:
            side_center = 90.0
            front_of_side_center = 67.0

        # Get side sector (perpendicular to wall)
        s0, s1 = self._indices_for_degs(angle_min, angle_inc, n,
                                        side_center - SIDE_DEG/2.0,
                                        side_center + SIDE_DEG/2.0)
        
        # Get front-of-side sector (for angle detection)
        f0, f1 = self._indices_for_degs(angle_min, angle_inc, n,
                                        front_of_side_center - FRONT_OF_SIDE_DEG/2.0,
                                        front_of_side_center + FRONT_OF_SIDE_DEG/2.0)

        side_vals = self._subset_ranges(ranges, s0, s1)
        front_vals = self._subset_ranges(ranges, f0, f1)

        d_side = self._nanmean(side_vals)
        d_front = self._nanmean(front_vals)

        if math.isnan(d_side):
            return math.nan

        # Use simple side reading as base
        result = d_side

        # Apply trigonometric correction only if beneficial
        if not math.isnan(d_front) and abs(d_front - d_side) > 0.03:
            delta = abs(front_of_side_center - side_center)
            theta = math.radians(delta)
            
            if theta > 1e-3:
                try:
                    # Calculate wall angle
                    phi = math.atan2(d_front - d_side, d_side * math.tan(theta))
                    d_perp = d_side * math.cos(phi)
                    
                    # Use correction if reasonable (within 25cm difference)
                    if self._valid(d_perp) and abs(d_perp - d_side) < 0.25:
                        result = d_perp
                except (ValueError, ZeroDivisionError):
                    pass

        return result

    def front_min_distance(self, scan: LaserScan) -> float:
        """Get minimum distance in front sector with smoothing."""
        n = len(scan.ranges)
        i0, i1 = self._indices_for_degs(scan.angle_min, scan.angle_increment, n,
                                        -FRONT_DEG/2.0, FRONT_DEG/2.0)
        vals = self._subset_ranges(list(scan.ranges), i0, i1)
        
        raw = min(vals) if vals else math.inf

        if self.front_min_filt is None or not math.isfinite(self.front_min_filt):
            self.front_min_filt = raw
        else:
            self.front_min_filt = ((1.0 - FRONT_SMOOTH_ALPHA) * self.front_min_filt + 
                                   FRONT_SMOOTH_ALPHA * raw)
        return self.front_min_filt

    def left_right_min(self, scan: LaserScan) -> Tuple[float, float]:
        """Get minimum distances on left and right sides."""
        ranges = list(scan.ranges)
        n = len(ranges)
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment

        L0, L1 = self._indices_for_degs(angle_min, angle_inc, n, 60.0, 120.0)
        R0, R1 = self._indices_for_degs(angle_min, angle_inc, n, -120.0, -60.0)

        left_vals = self._subset_ranges(ranges, L0, L1)
        right_vals = self._subset_ranges(ranges, R0, R1)

        lmin = min(left_vals) if left_vals else math.inf
        rmin = min(right_vals) if right_vals else math.inf
        return lmin, rmin

    # ========== Subsumption Architecture Control ==========
    
    def on_scan(self, scan: LaserScan):
        """
        Main callback implementing subsumption architecture.
        Higher priority behaviors subsume lower ones.
        """
        self.scan_meta = (scan.angle_min, scan.angle_increment, len(scan.ranges))
        
        d_front_min = self.front_min_distance(scan)
        
        twist = Twist()
        now = time.monotonic()

        # ===== PRIORITY 1: ESCAPE STATE (Highest Priority) =====
        if self.state == 'ESCAPE':
            turn_dir = self.last_turn_dir
            twist.linear.x = OA_SPIN
            twist.angular.z = turn_dir * (OA_TURN * ANGULAR_LIMIT)
            self.pub.publish(twist)

            if (self.escape_until is not None and 
                now >= self.escape_until and 
                d_front_min > OA_EXIT_CLEAR):
                self.state = 'NORMAL'
                self.pid.reset()
                self._maybe_log(f"✓ EXIT ESCAPE -> NORMAL (front={d_front_min:.2f}m)")
            return

        # ===== PRIORITY 2: EMERGENCY AVOIDANCE =====
        if d_front_min < OA_NEAR:
            lmin, rmin = self.left_right_min(scan)
            self.last_turn_dir = -1.0 if lmin < rmin else 1.0
            self.escape_until = now + ESCAPE_TIME
            self.state = 'ESCAPE'
            
            twist.linear.x = OA_SPIN
            twist.angular.z = self.last_turn_dir * (OA_TURN * ANGULAR_LIMIT)
            self.pub.publish(twist)
            self._maybe_log(f"⚠ ESCAPE! front={d_front_min:.2f}m "
                          f"turn={'CCW↺' if self.last_turn_dir>0 else 'CW↻'}")
            return

        # ===== PRIORITY 3: CAUTION STATE =====
        if d_front_min < OA_CAUTION or self.state == 'CAUTION':
            lmin, rmin = self.left_right_min(scan)
            
            bias = (rmin - lmin)
            norm = math.atan(bias) / (math.pi/2.0)

            if abs(norm) < BIAS_DEADBAND:
                turn_dir = self.last_turn_dir
            else:
                turn_dir = 1.0 if norm >= 0.0 else -1.0
                self.last_turn_dir = turn_dir

            twist.linear.x = OA_SLOW
            twist.angular.z = turn_dir * (OA_TURN * ANGULAR_LIMIT)
            self.pub.publish(twist)

            old_state = self.state
            self.state = 'CAUTION' if d_front_min < OA_EXIT_CLEAR else 'NORMAL'
            
            if old_state == 'CAUTION' and self.state == 'NORMAL':
                self.pid.reset()
                
            if self.scan_count % PRINT_EVERY == 0:
                self.get_logger().info(
                    f"⚡ CAUTION: front={d_front_min:.2f}m "
                    f"turn={'CCW↺' if turn_dir>0 else 'CW↻'} -> {self.state}")
            return

        # ===== PRIORITY 4: WALL FOLLOWING (Lowest Priority) =====
        dist = self.estimate_side_distance(scan)
        
        # Smooth filtering
        if self.filtered_dist is None:
            self.filtered_dist = dist if self._valid(dist) else None
        else:
            if self._valid(dist):
                self.filtered_dist = ((1.0 - SMOOTHING_ALPHA) * self.filtered_dist + 
                                     SMOOTHING_ALPHA * dist)

        # No wall detected
        if self.filtered_dist is None or not self._valid(self.filtered_dist):
            twist.linear.x = LINEAR_SPEED * 0.7
            twist.angular.z = 0.0
            self.pub.publish(twist)
            if self.scan_count % PRINT_EVERY == 0:
                self.get_logger().info("⊙ No wall - exploring")
            return

        # Track min/max distances seen
        self.min_dist_seen = min(self.min_dist_seen, self.filtered_dist)
        self.max_dist_seen = max(self.max_dist_seen, self.filtered_dist)

        # PID control
        e = DESIRED_DISTANCE - self.filtered_dist
        u = self.pid.step(e, now)
        u = clamp(u, -ANGULAR_LIMIT, ANGULAR_LIMIT)

        twist.linear.x = LINEAR_SPEED
        twist.angular.z = u
        self.pub.publish(twist)

        # Logging
        self.scan_count += 1
        if self.scan_count % PRINT_EVERY == 0:
            side = FOLLOW_SIDE.upper()
            self.get_logger().info(
                f"║ {side}-WALL dist={self.filtered_dist:.3f}m "
                f"(target={DESIRED_DISTANCE:.2f}) err={e:+.3f}m "
                f"ang={u:+.3f}r/s │ range:[{self.min_dist_seen:.2f}-{self.max_dist_seen:.2f}]m")

    def _maybe_log(self, msg: str):
        """Conditional logging."""
        self.scan_count += 1
        if self.scan_count % PRINT_EVERY == 0:
            self.get_logger().info(msg)

def main():
    rclpy.init()
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop = Twist()
        node.pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()