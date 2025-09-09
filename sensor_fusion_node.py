#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu, JointState, LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import TwistWithCovarianceStamped
import math


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Data storage
        self.imu_data = None
        self.odom_data = None
        self.joint_data = None
        
        # Fusion parameters
        self.imu_weight = 0.3
        self.odom_weight = 0.7
        
        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        # Publisher
        self.fused_twist_pub = self.create_publisher(TwistWithCovarianceStamped, '/sensor_fused_twist', 10)
        
        # Timer
        self.timer = self.create_timer(0.1, self.fusion_callback)
        
        self.get_logger().info('Sensor Fusion Node initialized')
        
    def imu_callback(self, msg):
        self.imu_data = msg
        
    def odom_callback(self, msg):
        self.odom_data = msg
        
    def joint_callback(self, msg):
        self.joint_data = msg
        
    def fusion_callback(self):
        """Sensor fusion işlemi (basit yaklaşım, Kalman benzeri)"""
        if not self.imu_data or not self.odom_data:
            return
            
        fused_msg = TwistWithCovarianceStamped()
        fused_msg.header.stamp = self.get_clock().now().to_msg()
        fused_msg.header.frame_id = 'base_footprint'
        
        # Linear velocity fusion
        odom_linear_x = self.odom_data.twist.twist.linear.x
        
        # IMU velocity (accel integrate)
        if hasattr(self, 'last_imu_time') and self.imu_data:
            dt = 0.1
            imu_velocity_x = getattr(self, 'integrated_velocity', 0) + self.imu_data.linear_acceleration.x * dt
            self.integrated_velocity = imu_velocity_x
        else:
            imu_velocity_x = odom_linear_x
            self.integrated_velocity = imu_velocity_x
            
        self.last_imu_time = self.get_clock().now()
        
        fused_msg.twist.twist.linear.x = (
            self.odom_weight * odom_linear_x +
            self.imu_weight * imu_velocity_x
        )
        
        # Angular velocity (IMU > odom)
        if self.imu_data.angular_velocity.z != 0:
            fused_msg.twist.twist.angular.z = self.imu_data.angular_velocity.z
        else:
            fused_msg.twist.twist.angular.z = self.odom_data.twist.twist.angular.z
            
        fused_msg.twist.covariance[0] = 0.1
        fused_msg.twist.covariance[35] = 0.05
        
        self.fused_twist_pub.publish(fused_msg)


class AdvancedSlamNode(Node):
    def __init__(self):
        super().__init__('advanced_slam_node')
        
        self.map_width = 200
        self.map_height = 200
        self.map_resolution = 0.05
        self.map_origin_x = -5.0
        self.map_origin_y = -5.0
        
        self.occupancy_map = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        self.hit_count = np.zeros((self.map_height, self.map_width))
        self.miss_count = np.zeros((self.map_height, self.map_width))
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/slam_map', 10)
        self.timer = self.create_timer(1.0, self.publish_map)
        
        self.get_logger().info('Advanced SLAM Node initialized')
        
    def world_to_map(self, world_x, world_y):
        map_x = int((world_x - self.map_origin_x) / self.map_resolution)
        map_y = int((world_y - self.map_origin_y) / self.map_resolution)
        return map_x, map_y
        
    def is_valid_map_coord(self, map_x, map_y):
        return 0 <= map_x < self.map_width and 0 <= map_y < self.map_height
        
    def laser_callback(self, msg):
        angle = msg.angle_min
        for range_val in msg.ranges:
            if range_val < msg.range_min or range_val > msg.range_max:
                angle += msg.angle_increment
                continue
                
            hit_x = self.robot_x + range_val * math.cos(self.robot_yaw + angle)
            hit_y = self.robot_y + range_val * math.sin(self.robot_yaw + angle)
            
            hit_map_x, hit_map_y = self.world_to_map(hit_x, hit_y)
            self.update_map_ray(
                int(self.robot_x / self.map_resolution) + self.map_width // 2,
                int(self.robot_y / self.map_resolution) + self.map_height // 2,
                hit_map_x, hit_map_y
            )
            angle += msg.angle_increment
            
    def update_map_ray(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        
        while True:
            if self.is_valid_map_coord(x, y) and (x != x1 or y != y1):
                self.miss_count[y, x] += 1
            if x == x1 and y == y1:
                if self.is_valid_map_coord(x, y):
                    self.hit_count[y, x] += 1
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
    def update_occupancy_probabilities(self):
        for i in range(self.map_height):
            for j in range(self.map_width):
                total_hits = self.hit_count[i, j]
                total_misses = self.miss_count[i, j]
                total_observations = total_hits + total_misses
                if total_observations > 0:
                    probability = total_hits / total_observations
                    if probability > 0.6:
                        self.occupancy_map[i, j] = 100
                    elif probability < 0.4:
                        self.occupancy_map[i, j] = 0
                    else:
                        self.occupancy_map[i, j] = -1
                else:
                    self.occupancy_map[i, j] = -1
                    
    def publish_map(self):
        self.update_occupancy_probabilities()
        map_msg = OccupancyGrid()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = float(self.map_resolution)
        map_msg.info.width = self.map_width
        map_msg.info.height = self.map_height
        map_msg.info.origin.position.x = float(self.map_origin_x)
        map_msg.info.origin.position.y = float(self.map_origin_y)
        map_msg.info.origin.orientation.w = 1.0
        map_msg.data = self.occupancy_map.flatten().tolist()
        self.map_pub.publish(map_msg)


def main_sensor_fusion(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


def main_advanced_slam(args=None):
    rclpy.init(args=args)
    node = AdvancedSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'slam':
        main_advanced_slam()
    else:
        main_sensor_fusion()
