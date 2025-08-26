from setuptools import setup
import os
from glob import glob

package_name = 'advanced_robot_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='simay',
    maintainer_email='simay@example.com',
    description='Advanced Robot Navigation Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kalman_filter_node = advanced_robot_nav.kalman_filter_node:main',
            'advanced_slam_node = advanced_robot_nav.advanced_slam_node:main',
            'path_planner_node = advanced_robot_nav.path_planner_node:main',
            'pure_pursuit_node = advanced_robot_nav.pure_pursuit_node:main',
            'sensor_fusion_node = advanced_robot_nav.sensor_fusion_node:main',
        ],
    },
)
