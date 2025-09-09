import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf_transformations, math, csv, os, time

def yaw_of(q):
    return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

def dist2(ax, ay, bx, by):
    dx, dy = ax-bx, ay-by
    return dx*dx + dy*dy

class TrackingEval(Node):
    def __init__(self):
        super().__init__('tracking_eval_node')
        self.tf = Buffer(cache_time=Duration(seconds=10.0))
        self.listener = TransformListener(self.tf, self)
        self.create_subscription(Path, '/planned_path', self.path_cb, 10)
        self.path = []
        self.last_print = self.get_clock().now()
        self.window_cte = []
        self.window_heading = []
        self.window_size = 100  # ~10 sn @10Hz
        # csv
        logdir = os.path.expanduser('~/robotics_control_ws')
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(logdir, f'tracking_log_{ts}.csv')
        self.csv = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv)
        self.writer.writerow(['t', 'x', 'y', 'yaw', 'cte', 'heading_err_deg', 'nearest_idx', 'progress_pct'])
        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info('TrackingEval started; logging to ' + self.csv_path)

    def path_cb(self, msg: Path):
        self.path = [(p.pose.position.x, p.pose.position.y,
                      yaw_of(p.pose.orientation)) for p in msg.poses]
        self.get_logger().info(f'Path received: {len(self.path)} points')

    def get_robot_in_map(self):
        try:
            tf = self.tf.lookup_transform('map', 'base_link', rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        yaw = yaw_of(tf.transform.rotation)
        return x, y, yaw

    def nearest_on_polyline(self, x, y):
        """en yakın noktanın segment projeksiyonu (indis, cte, başlık)"""
        if len(self.path) < 2: return None
        best = (0, float('inf'), 0.0, 0.0)  # (idx, cte, seg_yaw, s)
        for i in range(len(self.path)-1):
            x1,y1,_ = self.path[i]
            x2,y2,_ = self.path[i+1]
            vx, vy = x2-x1, y2-y1
            L2 = vx*vx + vy*vy
            if L2 == 0: 
                cte = math.sqrt(dist2(x,y,x1,y1))
                th  = math.atan2(vy, vx)
                if cte < best[1]: best = (i, cte, th, 0.0)
                continue
            # projeksiyon parametresi
            t = max(0.0, min(1.0, ((x-x1)*vx + (y-y1)*vy)/L2))
            px, py = x1 + t*vx, y1 + t*vy
            # signed CTE: sol (+) / sağ (-)
            cross = (x - x1)*vy - (y - y1)*vx
            sign = 1.0 if cross > 0 else -1.0
            cte = sign*math.sqrt(dist2(x,y,px,py))
            th = math.atan2(vy, vx)
            if abs(cte) < abs(best[1]):
                best = (i, cte, th, t)
        return best

    def tick(self):
        pose = self.get_robot_in_map()
        if pose is None or len(self.path) < 2:
            return
        x,y,yaw = pose
        i, cte, seg_yaw, t = self.nearest_on_polyline(x,y)
        # heading error [-pi,pi]
        dh = (seg_yaw - yaw + math.pi)%(2*math.pi) - math.pi
        # progress
        progress = (i + t) / max(1, len(self.path)-1) * 100.0

        # log pencere
        self.window_cte.append(abs(cte))
        self.window_heading.append(abs(dh))
        if len(self.window_cte) > self.window_size:
            self.window_cte.pop(0); self.window_heading.pop(0)

        tnow = self.get_clock().now().nanoseconds*1e-9
        self.writer.writerow([tnow, x, y, yaw, cte, math.degrees(dh), i, progress])

        # 1 sn'de bir özet yaz
        if (self.get_clock().now() - self.last_print).nanoseconds > 1e9:
            rms_cte = math.sqrt(sum(c*c for c in self.window_cte)/len(self.window_cte))
            mean_head = sum(self.window_heading)/len(self.window_heading)
            self.get_logger().info(
                f'CTE={cte:+.2f} m | hdg={math.degrees(dh):+.1f}° | '
                f'RMS_CTE(10s)={rms_cte:.2f} m | mean|hdg|(10s)={math.degrees(mean_head):.1f}° | prog={progress:5.1f}%')
            self.last_print = self.get_clock().now()

    def destroy_node(self):
        try:
            self.csv.flush(); self.csv.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    n = TrackingEval()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()
