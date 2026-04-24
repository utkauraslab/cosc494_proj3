#!/usr/bin/env python3

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team
# License: BSD 3-Clause.

# Modifiled to ROS2 by Fei Liu, University of Tennessee, 2026

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped


class PointClickedToReposition(Node):

    def __init__(self):
        super().__init__("point_clicked_to_reposition")

        # Subscriber
        self.sub = self.create_subscription(PointStamped, "/clicked_point", self.point_clicked_cb, 10)  # QoS depth

        # Publisher (private namespace equivalent to "~")
        self.pub = self.create_publisher(PoseStamped, "reposition", 10)

    def point_clicked_cb(self, msg):
        as_pose = PoseStamped()
        as_pose.header = msg.header
        as_pose.pose.position = msg.point
        as_pose.pose.orientation.w = 1.0

        self.pub.publish(as_pose)


def main(args=None):
    rclpy.init(args=args)
    node = PointClickedToReposition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
