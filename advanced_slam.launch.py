from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Paket dizinleri
    pkg_advanced_robot_nav = FindPackageShare('advanced_robot_nav')
    pkg_turtlebot3_gazebo = FindPackageShare('turtlebot3_gazebo')

    # TB3 URDF dosyasını doğrudan oku (xacro yok)
    tb3_desc_dir = get_package_share_directory('turtlebot3_description')
    model = os.environ.get('TURTLEBOT3_MODEL', 'waffle')
    if model not in ('burger', 'waffle', 'waffle_pi'):
        model = 'waffle'
    urdf_name = f"turtlebot3_{model}.urdf"
    urdf_path = os.path.join(tb3_desc_dir, 'urdf', urdf_name)
    with open(urdf_path, 'r') as f:
        robot_description_xml = f.read()

    # Argümanlar
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')
    world_arg        = DeclareLaunchArgument('world', default_value='turtlebot3_world')

    # Gazebo dünya
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_turtlebot3_gazebo, 'launch', 'turtlebot3_world.launch.py'])
        ]),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'robot_description': robot_description_xml
        }],
        output='screen'
    )

    # Cartographer SLAM - launch include yerine doğrudan nodlar
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=[
            '-configuration_directory', PathJoinSubstitution([pkg_advanced_robot_nav, 'config']),
            '-configuration_basename', 'turtlebot3_lds_2d.lua'
        ],
        output='screen'
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['-resolution', '0.05'],
        output='screen'
    )

    # Bizim nodelar
    kalman_filter_node = Node(
        package='advanced_robot_nav', executable='kalman_filter_node',
        name='kalman_filter_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )
    path_planner_node = Node(
        package='advanced_robot_nav', executable='path_planner_node',
        name='path_planner_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )
    pure_pursuit_node = Node(
        package='advanced_robot_nav', executable='pure_pursuit_node',
        name='pure_pursuit_node',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        output='screen'
    )

    # RViz
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_advanced_robot_nav, 'rviz', 'advanced_nav.rviz'])],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return LaunchDescription([
        use_sim_time_arg, world_arg,
        gazebo_launch, robot_state_publisher,
        cartographer_node, occupancy_grid_node,
        kalman_filter_node, path_planner_node, pure_pursuit_node, rviz_node
    ])
