#!/usr/bin/env python3

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

# Modifiled to ROS2 by Fei Liu, University of Tennessee, 2026


from __future__ import absolute_import, division, print_function

import numpy as np
import range_libc

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
from geometry_msgs.msg import Transform
from sensor_msgs.msg import LaserScan

from mushr_sim import utils


class FakeURG:
    def __init__(self, node: Node, map_msg, topic_namespace: str = ""):
        self.node = node

        # -------- Parameters --------
        # self.node.declare_parameter("update_rate", 10.0)
        self.node.declare_parameter("theta_discretization", 656.0)
        self.node.declare_parameter("min_range_meters", 0.02)
        self.node.declare_parameter("max_range_meters", 10.0)
        self.node.declare_parameter("angle_step", 0.00613592332229)
        self.node.declare_parameter("angle_min", -2.08621382713)
        self.node.declare_parameter("angle_max", 2.09234976768)
        self.node.declare_parameter("car_length", 0.33)

        self.node.declare_parameter("z_short", 0.03)
        self.node.declare_parameter("z_max", 0.16)
        self.node.declare_parameter("z_blackout_max", 50.0)
        self.node.declare_parameter("z_rand", 0.01)
        self.node.declare_parameter("z_hit", 0.8)
        self.node.declare_parameter("z_sigma", 0.03)

        # self.node.declare_parameter("tf_prefix", "")

        # Read params
        self.UPDATE_RATE = float(self.node.get_parameter("update_rate").value)
        self.THETA_DISCRETIZATION = float(self.node.get_parameter("theta_discretization").value)
        self.MIN_RANGE_METERS = float(self.node.get_parameter("min_range_meters").value)
        self.MAX_RANGE_METERS = float(self.node.get_parameter("max_range_meters").value)
        self.ANGLE_STEP = float(self.node.get_parameter("angle_step").value)
        self.ANGLE_MIN = float(self.node.get_parameter("angle_min").value)
        self.ANGLE_MAX = float(self.node.get_parameter("angle_max").value)
        self.ANGLES = np.arange(self.ANGLE_MIN, self.ANGLE_MAX, self.ANGLE_STEP, dtype=np.float32)

        self.CAR_LENGTH = float(self.node.get_parameter("car_length").value)

        self.Z_SHORT = float(self.node.get_parameter("z_short").value)
        self.Z_MAX = float(self.node.get_parameter("z_max").value)
        self.Z_BLACKOUT_MAX = int(float(self.node.get_parameter("z_blackout_max").value))
        self.Z_RAND = float(self.node.get_parameter("z_rand").value)
        self.Z_HIT = float(self.node.get_parameter("z_hit").value)
        self.Z_SIGMA = float(self.node.get_parameter("z_sigma").value)

        self.TF_PREFIX = str(self.node.get_parameter("tf_prefix").value).rstrip("/")
        if len(self.TF_PREFIX) > 0:
            self.TF_PREFIX = self.TF_PREFIX + "/"

        # -------- range_libc setup --------
        occ_map = range_libc.PyOMap(map_msg)
        max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution)
        self.range_method = range_libc.PyCDDTCast(occ_map, max_range_px, self.THETA_DISCRETIZATION)

        # -------- TF2 setup --------
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

        # Throttle helper
        self._last_tf_warn = None

        # Get base_link -> laser_link static offset (x_offset)
        self.x_offset = self._wait_for_laser_offset()

        # -------- Publisher --------
        # ROS1 used "~{namespace}scan". In ROS2 we publish to "{namespace}scan"
        topic = f"{topic_namespace}scan"
        self.laser_pub = self.node.create_publisher(LaserScan, topic, 1)

        # -------- Timer --------
        self.update_timer = self.node.create_timer(1.0 / self.UPDATE_RATE, self.timer_cb)

        # Start at 0,0,0 with identity rotation
        self.transform = Transform()
        self.transform.rotation.w = 1.0

    def _warn_throttle(self, period_sec: float, msg: str):
        now = self.node.get_clock().now()
        if self._last_tf_warn is None or (now - self._last_tf_warn) > Duration(seconds=period_sec):
            self._last_tf_warn = now
            self.node.get_logger().warn(msg)

    def _wait_for_laser_offset(self) -> float:
        # Equivalent to ROS1 loop with Rate sleep; here we retry with a timer-like loop
        target = self.TF_PREFIX + "laser_link"
        source = self.TF_PREFIX + "base_link"

        while rclpy.ok():
            try:
                # time=Time() means "latest available" in rclpy tf2
                tf_msg = self._tf_buffer.lookup_transform(source, target, rclpy.time.Time())  # target_frame (base_link)  # source_frame (laser_link)
                # We only need translation.x from base_link <- laser_link transform
                return float(tf_msg.transform.translation.x)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self._warn_throttle(5.0, f"Waiting for TF {source} <- {target}: {e}")
                # Let callbacks run a bit
                rclpy.spin_once(self.node, timeout_sec=0.1)

        # If shutdown, default 0
        return 0.0

    def noise_laser_scan(self, ranges: np.ndarray):
        indices = np.zeros(ranges.shape[0], dtype=np.int32)

        prob_sum = self.Z_HIT + self.Z_RAND + self.Z_SHORT
        hit_count = int((self.Z_HIT / prob_sum) * indices.shape[0])
        rand_count = int((self.Z_RAND / prob_sum) * indices.shape[0])
        short_count = indices.shape[0] - hit_count - rand_count

        indices[hit_count : hit_count + rand_count] = 1
        indices[hit_count + rand_count :] = 2
        np.random.shuffle(indices)

        hit_indices = indices == 0
        ranges[hit_indices] += np.random.normal(loc=0.0, scale=self.Z_SIGMA, size=hit_count)

        rand_indices = indices == 1
        ranges[rand_indices] = np.random.uniform(low=self.MIN_RANGE_METERS, high=self.MAX_RANGE_METERS, size=rand_count)

        short_indices = indices == 2
        ranges[short_indices] = np.random.uniform(low=self.MIN_RANGE_METERS, high=ranges[short_indices], size=short_count)

        max_count = (self.Z_MAX / (prob_sum + self.Z_MAX)) * ranges.shape[0]
        while max_count > 0:
            cur = np.random.randint(low=0, high=ranges.shape[0], size=1)
            blackout_count = np.random.randint(low=1, high=self.Z_BLACKOUT_MAX, size=1)
            while cur > 0 and cur < ranges.shape[0] and blackout_count > 0 and max_count > 0:
                if not np.isnan(ranges[cur]):
                    ranges[cur] = np.nan
                    cur += 1
                    blackout_count -= 1
                    max_count -= 1
                else:
                    break

    def timer_cb(self):
        now = self.node.get_clock().now()

        ls = LaserScan()
        ls.header.frame_id = self.TF_PREFIX + "laser_link"
        ls.header.stamp = now.to_msg()

        ls.angle_increment = self.ANGLE_STEP
        ls.angle_min = self.ANGLE_MIN
        ls.angle_max = self.ANGLE_MAX
        ls.range_min = self.MIN_RANGE_METERS
        ls.range_max = self.MAX_RANGE_METERS
        ls.intensities = []

        ranges = np.zeros(len(self.ANGLES), dtype=np.float32)

        laser_angle = utils.quaternion_to_angle(self.transform.rotation)
        laser_pose_x = self.transform.translation.x + self.x_offset * np.cos(laser_angle)
        laser_pose_y = self.transform.translation.y + self.x_offset * np.sin(laser_angle)

        range_pose = np.array((laser_pose_x, laser_pose_y, laser_angle), dtype=np.float32).reshape(1, 3)

        self.range_method.calc_range_repeat_angles(range_pose, self.ANGLES, ranges)
        self.noise_laser_scan(ranges)

        ls.ranges = ranges.tolist()
        self.laser_pub.publish(ls)
