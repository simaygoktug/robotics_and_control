import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from std_msgs.msg import Float64MultiArray
import tf2_ros
import tf2_geometry_msgs


class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        
        # Kalman Filter parametreleri (senin kodundaki gibi)
        self.state_dim = 4  # [x, y, vx, vy]
        self.measurement_dim = 2  # [x, y]
        
        # State vector [x, y, vx, vy]
        self.x = np.zeros(self.state_dim)
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 1.0
        
        # Process noise covariance
        self.Q = np.diag([0.01, 0.01, 0.1, 0.1])
        
        # Measurement noise covariance  
        self.R = np.diag([0.15, 0.15])
        
        # State transition matrix (constant velocity model)
        self.dt = 0.1
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
            
        # Publishers
        self.fused_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10)
        self.fused_twist_pub = self.create_publisher(
            TwistWithCovarianceStamped, '/fused_twist', 10)
        self.debug_pub = self.create_publisher(
            Float64MultiArray, '/kalman_debug', 10)
        
        # Timer for prediction step
        self.timer = self.create_timer(self.dt, self.predict_step)
        
        self.get_logger().info('Kalman Filter Node initialized')
        
        # Data storage
        self.imu_data = None
        self.odom_data = None
        
    def predict_step(self):
        """Kalman Filter Prediction Step"""
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.publish_fused_data()
        
    def update_step(self, measurement):
        """Kalman Filter Update Step"""
        # Innovation
        y = measurement - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
    def imu_callback(self, msg):
        """IMU verisi geldiğinde çalışır"""
        self.imu_data = msg
        
        # IMU'dan linear acceleration kullanarak velocity update
        if hasattr(msg, 'linear_acceleration'):
            self.x[2] += msg.linear_acceleration.x * self.dt
            self.x[3] += msg.linear_acceleration.y * self.dt
            
    def odom_callback(self, msg):
        """Odometry verisi geldiğinde çalışır"""
        self.odom_data = msg
        
        # Position measurement
        measurement = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])
        
        # Update step
        self.update_step(measurement)
        
    def publish_fused_data(self):
        """Fused pose ve twist yayınla"""
        # Fused Pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'odom'
        
        pose_msg.pose.pose.position.x = float(self.x[0])
        pose_msg.pose.pose.position.y = float(self.x[1])
        pose_msg.pose.pose.position.z = 0.0
        
        # Covariance matrix (6x6 for pose)
        pose_cov = np.zeros(36)
        pose_cov[0] = self.P[0, 0]  # x-x
        pose_cov[1] = self.P[0, 1]  # x-y  
        pose_cov[6] = self.P[1, 0]  # y-x
        pose_cov[7] = self.P[1, 1]  # y-y
        pose_msg.pose.covariance = pose_cov.tolist()
        
        self.fused_pose_pub.publish(pose_msg)
        
        # Fused Twist
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'odom'
        
        twist_msg.twist.twist.linear.x = float(self.x[2])
        twist_msg.twist.twist.linear.y = float(self.x[3])
        
        # Velocity covariance
        twist_cov = np.zeros(36)
        twist_cov[0] = self.P[2, 2]  # vx-vx
        twist_cov[7] = self.P[3, 3]  # vy-vy
        twist_msg.twist.covariance = twist_cov.tolist()
        
        self.fused_twist_pub.publish(twist_msg)
        
        # Debug data
        debug_msg = Float64MultiArray()
        debug_msg.data = [
            self.x[0], self.x[1], self.x[2], self.x[3],  # state
            self.P[0,0], self.P[1,1], self.P[2,2], self.P[3,3]  # diagonal covariance
        ]
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    kalman_filter = KalmanFilterNode()
    
    try:
        rclpy.spin(kalman_filter)
    except KeyboardInterrupt:
        pass
    
    kalman_filter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
