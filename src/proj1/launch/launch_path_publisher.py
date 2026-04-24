#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ---- Resolve default plan file path ----
    intro_pkg_share = get_package_share_directory("proj1")
    default_plan = os.path.join(intro_pkg_share, "plans", "shape_8.txt")

    # ---- Launch Arguments ----
    plan_file_arg = DeclareLaunchArgument(
        "plan_file",
        default_value=default_plan,
        description="Path to plan file",
    )

    buffer_size_arg = DeclareLaunchArgument(
        "buffer_size",
        default_value="200",
        description="Pose listener buffer size",
    )

    plan_file = LaunchConfiguration("plan_file")
    buffer_size = LaunchConfiguration("buffer_size")

    # ---- path_publisher node ----
    path_publisher_node = Node(
        package="proj1",
        executable="path_publisher",
        name="path_publisher",
        output="screen",
        parameters=[{"plan_file": plan_file}, {"use_sim_time": False}],
    )

    # ---- pose_listener node ----
    pose_listener_node = Node(
        package="proj1",
        executable="pose_listener",
        name="pose_listener",
        output="screen",
        parameters=[{"buffer_size": buffer_size}],
    )

    return LaunchDescription(
        [
            plan_file_arg,
            buffer_size_arg,
            path_publisher_node,
            pose_listener_node,
        ]
    )
