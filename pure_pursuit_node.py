import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Pose

import tf_transformations
from tf2_ros import Buffer, TransformListener, TransformException


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # ---- Params ----
        self.declare_parameter('lookahead_min', 0.5)
        self.declare_parameter('lookahead_max', 3.0)
        self.declare_parameter('v_max', 0.8)
        self.declare_parameter('w_max', 1.0)
        self.declare_parameter('goal_tol', 0.20)

        self.lookahead_min = float(self.get_parameter('lookahead_min').value)
        self.lookahead_max = float(self.get_parameter('lookahead_max').value)
        self.v_max         = float(self.get_parameter('v_max').value)
        self.w_max         = float(self.get_parameter('w_max').value)
        self.goal_tol      = float(self.get_parameter('goal_tol').value)

        # ---- State ----
        self.path: list[PoseStamped] | None = None
        self.target_idx = 0
        self.curr_lin_vel = 0.0
        self._last_warn_time = self.get_clock().now()

        # ---- TF (map -> base_link) ----
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- IO ----
        self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/pure_pursuit_target', 10)

        # ---- Control loop ----
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

        self.get_logger().info('Pure Pursuit Node (map frame) initialized')

    # ---- helpers ----
    def _warn_throttle(self, msg: str, period_sec: float = 2.0):
        now = self.get_clock().now()
        if (now - self._last_warn_time) > Duration(seconds=period_sec):
            self.get_logger().warning(msg)
            self._last_warn_time = now

    def path_cb(self, msg: Path):
        self.path = list(msg.poses)
        self.target_idx = 0
        self.get_logger().info(f'New path received with {len(self.path)} poses')

    def odom_cb(self, msg: Odometry):
        self.curr_lin_vel = abs(msg.twist.twist.linear.x)

    def get_robot_pose_in_map(self) -> Pose | None:
        # TF hazır mı?
        if not self.tf_buffer.can_transform('map', 'base_link', rclpy.time.Time(),
                                            timeout=Duration(seconds=0.5)):
            self._warn_throttle('Waiting for TF map->base_link… (SLAM henüz hazır olmayabilir)')
            return None
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        except TransformException as e:
            self._warn_throttle(f'TF error: {e}')
            return None

        p = Pose()
        p.position.x = tf.transform.translation.x
        p.position.y = tf.transform.translation.y
        p.position.z = tf.transform.translation.z
        p.orientation = tf.transform.rotation
        return p

    @staticmethod
    def dist(p1: Pose, p2: Pose) -> float:
        return math.hypot(p1.position.x - p2.position.x, p1.position.y - p2.position.y)

    @staticmethod
    def yaw(q) -> float:
        return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

    def find_target(self, robot: Pose) -> PoseStamped | None:
        if not self.path:
            return None

        # basit adaptif lookahead
        lookahead = max(self.lookahead_min,
                        min(self.lookahead_max, max(self.curr_lin_vel, 0.2) * 2.0))

        # en yakın index
        dmin, i0 = float('inf'), 0
        for i, ps in enumerate(self.path):
            d = self.dist(robot, ps.pose)
            if d < dmin:
                dmin, i0 = d, i

        # i0'dan ileri giderek lookahead'i aşan ilk nokta
        start = max(i0, self.target_idx)
        for i in range(start, len(self.path)):
            if self.dist(robot, self.path[i].pose) >= lookahead:
                self.target_idx = i
                return self.path[i]

        return self.path[-1]

    def control_loop(self):
        if not self.path or len(self.path) == 0:
            self._warn_throttle('No /planned_path yet…')
            return

        robot = self.get_robot_pose_in_map()
        if robot is None:
            return

        # goal kontrol
        if self.dist(robot, self.path[-1].pose) < self.goal_tol:
            self.cmd_pub.publish(Twist())
            self._warn_throttle('Goal reached.')
            return

        target = self.find_target(robot)
        if target is None:
            self._warn_throttle('No valid target on path.')
            return

        # viz
        target.header.frame_id = 'map'
        target.header.stamp = self.get_clock().now().to_msg()
        self.target_pub.publish(target)

        # pure pursuit
        dx = target.pose.position.x - robot.position.x
        dy = target.pose.position.y - robot.position.y
        yaw = self.yaw(robot.orientation)
        target_angle = math.atan2(dy, dx)
        angle_diff = (target_angle - yaw + math.pi) % (2 * math.pi) - math.pi

        ld = math.hypot(dx, dy)
        curvature = 2.0 * math.sin(angle_diff) / ld if ld > 1e-3 else 0.0

        cmd = Twist()
        # hız/denge
        if abs(angle_diff) > math.pi / 4:
            cmd.linear.x = 0.3 * self.v_max
        elif abs(angle_diff) > math.pi / 6:
            cmd.linear.x = 0.6 * self.v_max
        else:
            cmd.linear.x = self.v_max

        cmd.angular.z = max(-self.w_max, min(self.w_max, curvature * cmd.linear.x))
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
