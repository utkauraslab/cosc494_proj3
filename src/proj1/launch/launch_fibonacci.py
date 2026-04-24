#!/usr/bin/env python3

"""
You need to wrap all content inside generate_launch_description().

ROS2 launch files are written in Python (not XML like ROS1).
This file is the ROS2 rewrite of the original ROS1 XML launch skeleton.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    ROS launch files accept arguments using DeclareLaunchArgument.
    It allows you to pass arguments in from the command
    line (or from other launch files).
    """

    index_arg = DeclareLaunchArgument("index", default_value="10")
    fibonacci_output_topic_arg = DeclareLaunchArgument("fibonacci_output_topic", default_value="/proj1/fib_output")

    """
    To change the argument at run time from the command line:

        $ ros2 launch proj1 launch_fibonacci.py index:=20

    """

    # LaunchConfiguration is how we refer to launch arguments later
    # These arguments are the same as the ones we declared above with default values.
    # The default values will be replaced if we pass in arguments from the command line (such as ros2 run etc).
    index = LaunchConfiguration("index")
    fibonacci_output_topic = LaunchConfiguration("fibonacci_output_topic")

    """
    The Node object denotes a single ROS2 node to be launched.

    It needs:
       package    → where to find the executable for the node
       executable → name of the executable (for Python, from console_scripts in setup.py)
       name       → name of the node, so other nodes can refer to your node
    """

    fibonacci_node = Node(
        package="proj1",
        executable="fibonacci",
        name="fibonacci",
        output="screen",
        # Define the parameters for the node using the parameters field.
        # parameters is different from launch arguments:
        # - args are only used by the launch system to allow custom values
        # - parameters become available to the ROS system once it's running
        parameters=[
            {
                # Equivalent to:
                # <param name="output_topic" value="$(arg fibonacci_output_topic)" />
                "output_topic": fibonacci_output_topic,
                # Create a ROS parameter for the Fibonacci index,
                # using the launch file argument from above.
                # BEGIN QUESTION 1.3
                "index": index,
                # END QUESTION 1.3
            }
        ],
    )

    return LaunchDescription(
        [
            index_arg,
            fibonacci_output_topic_arg,
            fibonacci_node,
        ]
    )
