#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression, EnvironmentVariable
from launch.conditions import IfCondition
from launch_ros.actions import Node

import xacro


def robot_state_publisher_spawner(context: LaunchContext, xacro_file, car_name):
    xacro_path = context.perform_substitution(xacro_file)
    car_name_str = context.perform_substitution(car_name)

    robot_description = xacro.process_file(xacro_path).toprettyxml(indent="  ")

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
            remappings=[
                ("joint_states", f"/{car_name_str}/joint_states"),
            ],
        )
    ]


def generate_launch_description():

    mushr_desc_share = get_package_share_directory("mushr_description")
    mushr_sim_share = get_package_share_directory("mushr_sim")

    car_name = LaunchConfiguration("car_name")
    teleop = LaunchConfiguration("teleop")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_theta = LaunchConfiguration("initial_theta")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    xacro_file = LaunchConfiguration("xacro_file")
    map_file = LaunchConfiguration("map_file")

    default_xacro = os.path.join(mushr_desc_share, "robots", "mushr_nano.urdf.xacro")
    default_rviz = os.path.join(mushr_desc_share, "rviz", "default.rviz")
    default_map = os.path.join(mushr_sim_share, "maps", "maze_0.yaml")

    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{"yaml_filename": map_file}],
    )

    lifecycle_manager_map = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )

    mushr_sim_node = Node(
        package="mushr_sim",
        executable="mushr_sim",
        name="mushr_sim",
        output="screen",
        parameters=[
            {
                "car_name": car_name,
                "initial_x": initial_x,
                "initial_y": initial_y,
                "initial_theta": initial_theta,
                "map_service": "/map_server/map",
            }
        ],
    )

    keyboard_teleop = Node(
        package="mushr_sim",
        executable="keyboard_teleop_terminal",
        name="keyboard_teleop",
        output="screen",
        parameters=[
            {
                "car_name": car_name,
            }
        ],
        condition=IfCondition(teleop),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
    )

    ackermann_to_vesc = Node(
        package="vesc_ackermann",
        executable="ackermann_to_vesc_node",
        name="ackermann_to_vesc",
        namespace=car_name,
        output="screen",
        parameters=[
            {
                # Use the same calibration we used in mushr_sim.py
                "speed_to_erpm_gain": 4614.0,
                "speed_to_erpm_offset": 0.0,
                "steering_angle_to_servo_gain": -1.2135,
                "steering_angle_to_servo_offset": 0.5304,
            }
        ],
    )

    fake_vesc_driver = Node(
        package="mushr_sim",
        executable="fake_vesc_driver",
        name="fake_vesc_driver",
        output="screen",
        parameters=[
            {
                "car_name": car_name,
            }
        ],
    )

    fake_localization_node = Node(
        package="mushr_sim",
        executable="fake_localization",
        name="fake_localization",
        output="screen",
        parameters=[
            {"ground_truth_odom_topic": "/utk_car/odom"},
            {"map_frame": "map"},
            {"odom_frame": "odom"},
            {"base_frame": "base_footprint"},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("car_name", default_value="utk_car"),
            DeclareLaunchArgument("teleop", default_value="false"),
            DeclareLaunchArgument("initial_x", default_value="0.0"),
            DeclareLaunchArgument("initial_y", default_value="0.0"),
            DeclareLaunchArgument("initial_theta", default_value="0.0"),
            DeclareLaunchArgument("xacro_file", default_value=default_xacro),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz),
            DeclareLaunchArgument("map_file", default_value=default_map),
            # Spawn robot_state_publisher after substitutions are resolvable
            OpaqueFunction(function=robot_state_publisher_spawner, args=[xacro_file, car_name]),
            map_server_node,
            lifecycle_manager_map,
            mushr_sim_node,
            rviz_node,
            keyboard_teleop,
            ackermann_to_vesc,
            fake_vesc_driver,
            fake_localization_node,
        ]
    )
