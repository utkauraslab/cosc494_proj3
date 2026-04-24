#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

import numpy as np

from geometry_msgs.msg import Quaternion, TransformStamped

import tf_transformations  # pip/apt package in ROS2 (provides quaternion/euler helpers)


def angle_to_quaternion(angle: float) -> Quaternion:
    """Convert yaw angle (rad) to a Quaternion message."""
    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, float(angle))
    return Quaternion(x=qx, y=qy, z=qz, w=qw)


def quaternion_to_angle(q: Quaternion) -> float:
    """Convert a Quaternion message to yaw angle (rad)."""
    roll, pitch, yaw = tf_transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
    return float(yaw)


def map_to_world(pose, map_info):
    """
    Convert pose in map/grid coordinates (pixels) to world coordinates (meters).
    pose: array-like (x, y, theta)
    map_info: nav_msgs/MapMetaData
    """
    scale = float(map_info.resolution)
    angle = quaternion_to_angle(map_info.origin.orientation)

    pose = np.asarray(pose, dtype=np.float64)
    world_pose = np.zeros_like(pose, dtype=np.float64)

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    world_pose[0] = c * pose[0] - s * pose[1]
    world_pose[1] = s * pose[0] + c * pose[1]

    # scale
    world_pose[0] *= scale
    world_pose[1] *= scale

    # translate + angle
    world_pose[0] += float(map_info.origin.position.x)
    world_pose[1] += float(map_info.origin.position.y)
    world_pose[2] = float(pose[2]) + float(angle)

    return world_pose


def world_to_map(pose, map_info):
    """
    Convert pose in world coordinates (meters) to map/grid coordinates (pixels).
    pose: array-like (x, y, theta)
    map_info: nav_msgs/MapMetaData
    """
    scale = float(map_info.resolution)
    angle = -quaternion_to_angle(map_info.origin.orientation)

    map_pose = np.array(pose, dtype=np.float64)

    # translation
    map_pose[0] -= float(map_info.origin.position.x)
    map_pose[1] -= float(map_info.origin.position.y)

    # scale
    map_pose[0] *= 1.0 / scale
    map_pose[1] *= 1.0 / scale

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    tmp = map_pose[0]
    map_pose[0] = c * map_pose[0] - s * map_pose[1]
    map_pose[1] = s * tmp + c * map_pose[1]
    map_pose[2] = float(map_pose[2]) + float(angle)

    return map_pose


def make_transform_msg(node, translation, rotation, to_frame: str, from_frame: str) -> TransformStamped:
    """
    Create a TransformStamped.
    In ROS2, timestamps come from node.get_clock().now().

    node: rclpy.node.Node
    translation: array-like [x, y] (meters)
    rotation: yaw (rad)
    to_frame: child frame id
    from_frame: header frame id (parent)
    """
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id = from_frame
    t.child_frame_id = to_frame

    t.transform.translation.x = float(translation[0])
    t.transform.translation.y = float(translation[1])
    t.transform.translation.z = 0.0

    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, float(rotation))
    t.transform.rotation.x = float(qx)
    t.transform.rotation.y = float(qy)
    t.transform.rotation.z = float(qz)
    t.transform.rotation.w = float(qw)

    return t
