#!/usr/bin/env python3

"""
Tip for students:
  - In ROS2, use:
      ros2 topic list
      ros2 topic info /some_topic
    to discover the message type and topic name.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from proj1 import utils

# BEGIN QUESTION 2.3
# In ROS2, you typically do:
#   from some_msgs.msg import SomeMsg
#
# Student task: figure out the pose topic type (via `ros2 topic info <topic>`),
# then import the message type here.

# END QUESTION 2.3


def norm_python(data):
    """Compute the norm for each row of a numpy array using Python for loops.

    >>> data = np.array([[3, 4],
    ...                  [5, 12]])
    >>> norm_python(data)
    array([ 5., 13.])
    """
    n, d = data.shape
    norm = np.zeros(n)
    # BEGIN QUESTION 2.1
    # Student task: compute sqrt(sum_j data[i,j]^2) for each row i using loops.

    # END QUESTION 2.1
    return norm


def norm_numpy(data):
    """Compute the norm for each row of a numpy array using numpy functions.

    >>> data = np.array([[3, 4],
    ...                  [5, 12]])
    >>> norm_numpy(data)
    array([ 5., 13.])
    """
    # You can call np.sqrt, np.sum, np.square, etc.
    # Hint: you may find the `axis` parameter useful.
    # BEGIN QUESTION 2.2
    # Student task: do this without Python loops.

    # END QUESTION 2.2


class PoseListener(Node):
    """Collect car poses."""

    def __init__(self, size=100):
        # NOTE: In ROS2, node names must be unique in the graph.
        # If you run this script twice at the same time with the same node name,
        # you may see warnings about duplicate log publishers.
        super().__init__("pose_listener")

        self.size = size
        self.done = False
        self.storage = []  # a list of (x, y) tuples

        # Create a subscriber for the car pose.
        # Hint: once you've figured out the right message type, don't forget to
        # import it at the top! If the message type from `ros2 topic info` is
        # "X_msgs/msg/Y", the Python import would be "from X_msgs.msg import Y".
        #
        # BEGIN QUESTION 2.3
        # Student task: choose the correct topic name and message type.

        # END QUESTION 2.3

    def callback(self, msg):
        """Store the x and y coordinates of the car."""
        """Bonus: store the orientation as well!"""
        header = msg.header

        # ROS2 timestamps have `sec`/`nanosec` (ROS1 used `secs`/`nsecs`)
        # self.get_logger().info("Received a new message with timestamp " + str(header.stamp.sec) + "(s)")
        self.get_logger().info("Received a new message. Now total messages: " + str(len(self.storage)))

        # Extract and store the x, y position from the message data,
        # Bonus: store the orientation as well!
        # BEGIN QUESTION 2.4
        # Student task: extract x,y, \theta from the pose message and append to storage.
        # Hint: the message has a `pose` field, which has `position` and `orientation` fields.
        # The position has `x` and `y` fields.
        # The orientation is a quaternion (with `x`, `y`, `z`, `w` fields). To convert the
        # quaternion to a single angle \theta, you can use the utlity function
        # `utils.quaternion_to_angle`. The utils is already imported "from proj1 import utils".
        # You can check out the code in proj1/utils.py for how to use it.

        # END QUESTION 2.4

        # Use >= (instead of ==) for robustness if messages arrive faster than expected.
        if len(self.storage) >= self.size:
            self.done = True
            self.get_logger().info("Received enough samples, destroying subscription")
            # ROS2 equivalent of rospy Subscriber.unregister()
            self.destroy_subscription(self.subscriber)


def main(args=None):
    """Run the PoseListener node and plot the resulting poses.

    This script plots the car's location and distance from the origin over time.
    """
    rclpy.init(args=args)

    # IMPORTANT: Use only ONE node in this script.
    # Creating an extra Node() (e.g., a separate "runner") is unnecessary and can
    # lead to confusion or duplicate node-name/logging issues.
    #
    # We keep PoseListener as the single node that:
    #   - declares/reads parameters
    #   - subscribes to the pose topic
    #   - logs status
    listener = PoseListener()

    # In ROS2 we declare a parameter.
    # You can set it in launch or CLI, e.g.:
    #   ros2 run <pkg> <exe> --ros-args -p buffer_size:=500
    listener.declare_parameter("buffer_size", 500)
    buffer_size = int(listener.get_parameter("buffer_size").value)

    assert buffer_size > 100, "Choose a buffer size at least 100."
    listener.get_logger().info("Creating listener with buffer size " + str(buffer_size))

    # Update the listener's buffer size (stop condition)
    listener.size = int(buffer_size)

    # Spin until we collect enough samples.
    # We emulate rospy.Rate(5) with a simple loop + spin_once.
    try:
        while rclpy.ok() and not listener.done:
            rclpy.spin_once(listener, timeout_sec=0.1)
            time.sleep(0.2)  # ~5 Hz
    except KeyboardInterrupt:
        pass

    locations = np.array(listener.storage)

    # Plot the locations
    plt.figure()
    plt.plot(locations[:, 0], locations[:, 1])
    plt.title("Car Location")
    plt.xlabel("Distance (m)")
    plt.ylabel("Distance (m)")
    plt.axis("equal")
    # By default, the figure will be saved to the current directory; modify this
    # line to save it elsewhere.
    plt.savefig("trajectory.png")
    # Uncomment plt.show() to visualize the plot
    # plt.show()

    # Use norm_numpy to compute the distance from the origin over time. Then,
    # plot the distance over time as a line chart and save the plot.
    # BEGIN QUESTION 2.5
    # Student task: compute distance over time, plot it, and save the plot.

    # END QUESTION 2.5

    # ==============================================
    # BEGIN Bonus QUESTION 1

    # END Bonus QUESTION 1

    # ==============================================
    # BEGIN Bonus QUESTION 2
    # Express the entire trajectory in the coordinate frame of the pose of the car at the
    # 100th sample (index 100). Plot the transformed trajectory and save it.
    #
    # Hint Steps:
    #   1. Translate all positions so the initial pose is at the origin.
    #   2. Rotate all positions by -theta0 so the initial heading is aligned with x-axis.
    #   3. Compute heading relative to initial heading.
    #   4. Plot the transformed trajectory and save it.

    # END Bonus QUESTION 2

    listener.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
