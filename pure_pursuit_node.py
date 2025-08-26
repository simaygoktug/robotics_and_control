import rclpy
from rclpy.node import Node
import numpy as np
import math
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Float64
import tf_transformations


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Pure Pursuit parametreleri
        self.lookahead_distance = 1.0  # Lookahead distance
        self.min_lookahead = 0.5
        self.max_lookahead = 3.0
        self.max_linear_velocity = 0.8
        self.max_angular_velocity = 1.0
        self.goal_tolerance = 0.2
        
        # Robot state
        self.current_pose = None
        self.current_path = None
        self.target_index = 0
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.fused_pose_sub = self.create_subscription(
            PoseStamped, '/fused_pose', self.fused_pose_callback, 10)
            
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/pure_pursuit_target', 10)
        self.lookahead_pub = self.create_publisher(Float64, '/lookahead_distance', 10)
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        
        self.get_logger().info('Pure Pursuit Node initialized')
        
    def path_callback(self, msg):
        """Path mesajı geldiğinde"""
        self.current_path = msg.poses
        self.target_index = 0
        self.get_logger().info(f'New path received with {len(msg.poses)} points')
        
    def odom_callback(self, msg):
        """Odometry callback"""
        self.current_pose = msg.pose.pose
        
    def fused_pose_callback(self, msg):
        """Kalman filtered pose callback (daha doğru)"""
        self.current_pose = msg.pose
        
    def calculate_distance(self, pose1, pose2):
        """İki pose arası mesafe hesapla"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx**2 + dy**2)
        
    def get_yaw_from_quaternion(self, quaternion):
        """Quaternion'dan yaw açısı çıkar"""
        return tf_transformations.euler_from_quaternion([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        ])[2]
        
    def find_target_point(self):
        """Pure pursuit target noktası bul (senin kodundaki gibi)"""
        if not self.current_path or not self.current_pose:
            return None
            
        # Adaptive lookahead distance based on velocity
        current_velocity = getattr(self, 'current_velocity', 0.5)
        self.lookahead_distance = max(self.min_lookahead, 
                                    min(self.max_lookahead, 
                                        current_velocity * 2.0))
        
        # Find the closest point on path first
        min_distance = float('inf')
        closest_index = 0
        
        for i, pose_stamped in enumerate(self.current_path):
            distance = self.calculate_distance(self.current_pose, pose_stamped.pose)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
                
        # Start searching from closest point
        start_index = max(closest_index, self.target_index)
        
        # Find target point within lookahead distance
        for i in range(start_index, len(self.current_path)):
            distance = self.calculate_distance(self.current_pose, self.current_path[i].pose)
            
            if distance >= self.lookahead_distance:
                self.target_index = i
                return self.current_path[i]
                
        # If no point found, return last point
        if len(self.current_path) > 0:
            return self.current_path[-1]
            
        return None
        
    def pure_pursuit_control(self, target_pose):
        """Pure Pursuit kontrol algoritması"""
        if not self.current_pose:
            return Twist()
            
        # Calculate relative position
        dx = target_pose.pose.position.x - self.current_pose.position.x
        dy = target_pose.pose.position.y - self.current_pose.position.y
        
        # Current robot orientation
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        
        # Target angle
        target_angle = math.atan2(dy, dx)
        
        # Angle difference
        angle_diff = target_angle - current_yaw
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Distance to target
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        # Pure pursuit steering angle calculation
        # ld = lookahead distance, alpha = angle to target
        alpha = angle_diff
        ld = distance_to_target
        
        # Steering angle (simplified bicycle model)
        if ld > 0.01:  # Avoid division by zero
            curvature = 2 * math.sin(alpha) / ld
        else:
            curvature = 0
            
        # Create control command
        cmd = Twist()
        
        # Linear velocity - slow down for sharp turns
        if abs(angle_diff) > math.pi/4:  # 45 degrees
            cmd.linear.x = self.max_linear_velocity * 0.3
        elif abs(angle_diff) > math.pi/6:  # 30 degrees
            cmd.linear.x = self.max_linear_velocity * 0.6
        else:
            cmd.linear.x = self.max_linear_velocity
            
        # Angular velocity
        cmd.angular.z = curvature * cmd.linear.x
        cmd.angular.z = max(-self.max_angular_velocity, 
                           min(self.max_angular_velocity, cmd.angular.z))
        
        return cmd
        
    def control_loop(self):
        """Ana kontrol döngüsü"""
        if not self.current_path or not self.current_pose:
            return
            
        # Check if we reached the goal
        if len(self.current_path) > 0:
            goal_pose = self.current_path[-1]
            distance_to_goal = self.calculate_distance(self.current_pose, goal_pose.pose)
            
            if distance_to_goal < self.goal_tolerance:
                # Goal reached, stop the robot
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)
                self.get_logger().info('Goal reached!')
                return
                
        # Find target point
        target_pose = self.find_target_point()
        if not target_pose:
            return
            
        # Calculate control command
        cmd = self.pure_pursuit_control(target_pose)
        
        # Publish command
        self.cmd_vel_pub.publish(cmd)
        
        # Publish target point for visualization
        target_msg = PoseStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = 'map'
        target_msg.pose = target_pose.pose
        self.target_pub.publish(target_msg)
        
        # Publish lookahead distance
        lookahead_msg = Float64()
        lookahead_msg.data = self.lookahead_distance
        self.lookahead_pub.publish(lookahead_msg)


def main(args=None):
    rclpy.init(args=args)
    pure_pursuit = PurePursuitNode()
    
    try:
        rclpy.spin(pure_pursuit)
    except KeyboardInterrupt:
        pass
        
    pure_pursuit.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
