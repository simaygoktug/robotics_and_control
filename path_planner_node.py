import rclpy
from rclpy.node import Node
import numpy as np
import heapq
import random
import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        
        # QoS profil
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.BEST_EFFORT
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.start_sub = self.create_subscription(
            PoseStamped, '/initialpose', self.start_callback, 10)
            
        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.rrt_path_pub = self.create_publisher(Path, '/rrt_path', 10)
        
        # Map data
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.0
        self.map_origin = None
        
        # Start and goal positions
        self.start_pos = None
        self.goal_pos = None
        
        self.get_logger().info('Path Planner Node initialized')
        
    def map_callback(self, msg):
        """Map verisi geldiğinde"""
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        
        self.get_logger().info(f'Map received: {self.map_width}x{self.map_height}')
        
    def start_callback(self, msg):
        """Start position callback"""
        self.start_pos = self.world_to_grid(
            msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'Start position set: {self.start_pos}')
        self.plan_path()
        
    def goal_callback(self, msg):
        """Goal position callback"""
        self.goal_pos = self.world_to_grid(
            msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'Goal position set: {self.goal_pos}')
        self.plan_path()
        
    def world_to_grid(self, world_x, world_y):
        """World koordinatlarını grid koordinatlarına çevir"""
        if self.map_origin is None:
            return None
            
        grid_x = int((world_x - self.map_origin.position.x) / self.map_resolution)
        grid_y = int((world_y - self.map_origin.position.y) / self.map_resolution)
        
        return (grid_x, grid_y)
        
    def grid_to_world(self, grid_x, grid_y):
        """Grid koordinatlarını world koordinatlarına çevir"""
        world_x = grid_x * self.map_resolution + self.map_origin.position.x
        world_y = grid_y * self.map_resolution + self.map_origin.position.y
        return (world_x, world_y)
        
    def is_valid_position(self, x, y):
        """Pozisyon geçerli mi kontrol et"""
        if x < 0 or x >= self.map_width or y < 0 or y >= self.map_height:
            return False
        if self.map_data[y, x] > 50:  # Obstacle threshold
            return False
        return True
        
    def heuristic(self, a, b):
        """A* için heuristic function (senin kodundaki gibi)"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
    def astar_planning(self, start, goal):
        """A* Path Planning Algorithm"""
        if not self.is_valid_position(start[0], start[1]) or not self.is_valid_position(goal[0], goal[1]):
            self.get_logger().warn('Start or goal position is not valid')
            return []
            
        open_list = []
        heapq.heappush(open_list, (0 + self.heuristic(start, goal), 0, start, []))
        closed_set = set()
        
        # 8-connected grid
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        while open_list:
            f_cost, g_cost, current, path = heapq.heappop(open_list)
            
            if current == goal:
                return path + [goal]
                
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.is_valid_position(neighbor[0], neighbor[1]):
                    continue
                    
                if neighbor in closed_set:
                    continue
                
                # Cost calculation (diagonal moves cost more)
                move_cost = 1.0 if abs(dx) + abs(dy) == 1 else 1.414
                new_g_cost = g_cost + move_cost
                h_cost = self.heuristic(neighbor, goal)
                f_cost = new_g_cost + h_cost
                
                heapq.heappush(open_list, (f_cost, new_g_cost, neighbor, path + [current]))
                
        self.get_logger().warn('A* planning failed')
        return []
        
    def rrt_planning(self, start, goal, max_iterations=1000):
        """RRT Path Planning Algorithm"""
        if not self.is_valid_position(start[0], start[1]) or not self.is_valid_position(goal[0], goal[1]):
            return []
            
        nodes = [start]
        parent = {start: None}
        step_size = 10
        
        for i in range(max_iterations):
            # Random point
            if random.random() < 0.1:  # Goal bias
                rand_point = goal
            else:
                rand_point = (
                    random.randint(0, self.map_width-1),
                    random.randint(0, self.map_height-1)
                )
            
            # Find nearest node
            nearest_node = min(nodes, key=lambda n: self.heuristic(n, rand_point))
            
            # New node direction
            direction = (rand_point[0] - nearest_node[0], rand_point[1] - nearest_node[1])
            distance = math.sqrt(direction[0]**2 + direction[1]**2)
            
            if distance == 0:
                continue
                
            # Normalize and scale
            direction = (direction[0] / distance * step_size, direction[1] / distance * step_size)
            new_node = (
                int(nearest_node[0] + direction[0]), 
                int(nearest_node[1] + direction[1])
            )
            
            if not self.is_valid_position(new_node[0], new_node[1]):
                continue
                
            # Check collision along the path
            if self.check_collision_free(nearest_node, new_node):
                nodes.append(new_node)
                parent[new_node] = nearest_node
                
                # Check if we reached the goal
                if self.heuristic(new_node, goal) < step_size:
                    parent[goal] = new_node
                    
                    # Reconstruct path
                    path = []
                    current = goal
                    while current is not None:
                        path.append(current)
                        current = parent[current]
                    return path[::-1]
                    
        self.get_logger().warn('RRT planning failed')
        return []
        
    def check_collision_free(self, start, end):
        """İki nokta arası collision check"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return True
            
        steps = int(distance)
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(start[0] + t * dx)
            y = int(start[1] + t * dy)
            
            if not self.is_valid_position(x, y):
                return False
                
        return True
        
    def plan_path(self):
        """Path planning'i başlat"""
        if self.map_data is None or self.start_pos is None or self.goal_pos is None:
            return
            
        self.get_logger().info('Planning path...')
        
        # A* planning
        astar_path = self.astar_planning(self.start_pos, self.goal_pos)
        if astar_path:
            self.publish_path(astar_path, self.path_pub, 'A*')
            
        # RRT planning  
        rrt_path = self.rrt_planning(self.start_pos, self.goal_pos)
        if rrt_path:
            self.publish_path(rrt_path, self.rrt_path_pub, 'RRT')
            
    def publish_path(self, grid_path, publisher, algorithm_name):
        """Path'i ROS message olarak yayınla"""
        if not grid_path:
            return
            
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for grid_point in grid_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            
            world_x, world_y = self.grid_to_world(grid_point[0], grid_point[1])
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0
            
            # Orientation (simple forward direction)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose)
            
        publisher.publish(path_msg)
        self.get_logger().info(f'{algorithm_name} path published with {len(grid_path)} points')


def main(args=None):
    rclpy.init(args=args)
    path_planner = PathPlannerNode()
    
    try:
        rclpy.spin(path_planner)
    except KeyboardInterrupt:
        pass
        
    path_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
