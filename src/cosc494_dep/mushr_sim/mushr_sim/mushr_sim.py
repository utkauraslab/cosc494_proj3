#!/usr/bin/env python3
# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.
#
# Modified to ROS2 by Fei Liu, University of Tennessee, 2026
#
# Notes (ROS2):
# - Topics are made consistent with a configurable `car_name` parameter.
# - Nav2 map_server is lifecycle-managed; launch should activate it (nav2_lifecycle_manager) OR otherwise ensure
#   the map GetMap service is available.
# - This node publishes joint states to `/{car_name}/joint_states` (by default car="car").
#   If you want robot_state_publisher to consume these, remap robot_state_publisher's "joint_states" to that topic
#   (or publish to global "joint_states" instead).

from __future__ import absolute_import, division, print_function

from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
from tf2_ros import TransformBroadcaster, TransformListener

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped

from mushr_sim import utils
from mushr_sim.fake_urg import FakeURG
from mushr_interfaces.srv import CarPose

import pdb


class MushrSim(Node):
    """
    Publishes joint and TF information about the racecar.
    """

    def __init__(self):
        super().__init__("mushr_sim")

        # -------- Parameters (declare + read) --------
        # VESC conversions
        self.declare_parameter("vesc/speed_to_erpm_offset", 0.0)
        self.declare_parameter("vesc/speed_to_erpm_gain", 4614.0)
        self.declare_parameter("vesc/steering_angle_to_servo_offset", 0.5304)
        self.declare_parameter("vesc/steering_angle_to_servo_gain", -1.2135)
        self.declare_parameter("vesc/chassis_length", 0.33)
        self.declare_parameter("vesc/wheelbase", 0.25)

        # Noise + update rate
        self.declare_parameter("update_rate", 20.0)
        self.declare_parameter("speed_offset", 0.00)
        self.declare_parameter("speed_noise", 0.0001)
        self.declare_parameter("steering_angle_offset", 0.00)
        self.declare_parameter("steering_angle_noise", 0.000001)
        self.declare_parameter("forward_offset", 0.0)
        self.declare_parameter("forward_fix_noise", 0.0000001)
        self.declare_parameter("forward_scale_noise", 0.001)
        self.declare_parameter("side_offset", 0.0)
        self.declare_parameter("side_fix_noise", 0.000001)
        self.declare_parameter("side_scale_noise", 0.001)
        self.declare_parameter("theta_offset", 0.0)
        self.declare_parameter("theta_fix_noise", 0.000001)
        self.declare_parameter("use_mocap", False)

        # Initial pose
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_z", 0.0)  # unused in logic, kept for parity
        self.declare_parameter("initial_theta", 0.0)

        # Namespacing
        self.declare_parameter("car_name", "car")
        self.declare_parameter("tf_prefix", "")

        # Map service (Nav2 default is usually /map_server/map)
        self.declare_parameter("map_service", "/map_server/map")

        # Backward-compat (kept)
        self.declare_parameter("static_map", "static_map")

        # -------- Read params --------
        self.SPEED_TO_ERPM_OFFSET = float(self.get_parameter("vesc/speed_to_erpm_offset").value)
        self.SPEED_TO_ERPM_GAIN = float(self.get_parameter("vesc/speed_to_erpm_gain").value)

        self.STEERING_TO_SERVO_OFFSET = float(self.get_parameter("vesc/steering_angle_to_servo_offset").value)
        self.STEERING_TO_SERVO_GAIN = float(self.get_parameter("vesc/steering_angle_to_servo_gain").value)

        self.CAR_LENGTH = float(self.get_parameter("vesc/chassis_length").value)
        self.CAR_WIDTH = float(self.get_parameter("vesc/wheelbase").value)
        self.CAR_WHEEL_RADIUS = 0.0976 / 2.0

        self.UPDATE_RATE = float(self.get_parameter("update_rate").value)

        self.SPEED_OFFSET = float(self.get_parameter("speed_offset").value)
        self.SPEED_NOISE = float(self.get_parameter("speed_noise").value)

        self.STEERING_ANGLE_OFFSET = float(self.get_parameter("steering_angle_offset").value)
        self.STEERING_ANGLE_NOISE = float(self.get_parameter("steering_angle_noise").value)

        self.FORWARD_OFFSET = float(self.get_parameter("forward_offset").value)
        self.FORWARD_FIX_NOISE = float(self.get_parameter("forward_fix_noise").value)
        self.FORWARD_SCALE_NOISE = float(self.get_parameter("forward_scale_noise").value)

        self.SIDE_OFFSET = float(self.get_parameter("side_offset").value)
        self.SIDE_FIX_NOISE = float(self.get_parameter("side_fix_noise").value)
        self.SIDE_SCALE_NOISE = float(self.get_parameter("side_scale_noise").value)

        self.THETA_OFFSET = float(self.get_parameter("theta_offset").value)
        self.THETA_FIX_NOISE = float(self.get_parameter("theta_fix_noise").value)

        self.USE_MOCAP = bool(self.get_parameter("use_mocap").value)

        initial_x = float(self.get_parameter("initial_x").value)
        initial_y = float(self.get_parameter("initial_y").value)
        initial_theta = float(self.get_parameter("initial_theta").value)

        # self.CAR_NAME = self.get_parameter("car_name").get_parameter_value().string_value.strip() or "car"
        self.CAR_NAME = self.get_parameter("car_name").get_parameter_value().string_value.strip()

        self.TF_PREFIX = str(self.get_parameter("tf_prefix").value).rstrip("/")
        if len(self.TF_PREFIX) > 0:
            self.TF_PREFIX = self.TF_PREFIX + "/"

        # -------- Map --------
        self.permissible_region = None
        self.map_info = None

        self.map_service_name = self.get_parameter("map_service").get_parameter_value().string_value.strip()
        if not self.map_service_name.startswith("/"):
            self.map_service_name = "/" + self.map_service_name

        self.map_client = self.create_client(GetMap, self.map_service_name)
        if not self.map_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f"Map service '{self.map_service_name}' not available")

        self.permissible_region, self.map_info, raw_map_msg = self.get_map()

        # -------- State --------
        self.last_stamp = None  # rclpy Time
        self.last_speed = 0.0
        self.last_speed_lock = Lock()

        self.last_steering_angle = 0.0
        self.last_steering_angle_lock = Lock()

        self.cur_odom_to_base_trans = np.array([initial_x, initial_y], dtype=np.float64)
        self.cur_odom_to_base_rot = float(initial_theta)
        self.cur_odom_to_base_lock = Lock()

        self.cur_map_to_odom_trans = np.array([0.0, 0.0], dtype=np.float64)
        self.cur_map_to_odom_rot = 0.0
        self.cur_map_to_odom_lock = Lock()

        # -------- Joint message --------
        self.joint_msg = JointState()
        # IMPORTANT: These names must match the URDF joint names exactly.
        self.joint_msg.name = [
            "front_left_wheel_throttle",
            "front_right_wheel_throttle",
            "back_left_wheel_throttle",
            "back_right_wheel_throttle",
            "front_left_wheel_steer",
            "front_right_wheel_steer",
        ]
        self.joint_msg.position = [0.0] * 6

        # -------- TF --------
        self.br = TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- Fake laser --------
        # Ensure FakeURG publishes under the same car namespace.
        self.fake_laser = FakeURG(self, raw_map_msg, topic_namespace=f"{self.CAR_NAME}/")

        # -------- Publishers/Subscribers --------
        # Publish under /{car_name}/...
        if not self.USE_MOCAP:
            self.state_pub = self.create_publisher(PoseStamped, f"{self.CAR_NAME}/car_pose", 1)

        self.odom_pub = self.create_publisher(Odometry, f"{self.CAR_NAME}/odom", 1)
        self.cur_joints_pub = self.create_publisher(JointState, f"{self.CAR_NAME}/joint_states", 1)

        # Reposition inputs are kept un-namespaced unless you want multi-robot; adjust as needed.
        self.init_pose_sub = self.create_subscription(PoseStamped, "reposition", self.init_pose_cb, 1)

        # Subscribe using the same car namespace (these were hardcoded to /car/... previously)
        self.speed_sub = self.create_subscription(VescStateStamped, f"/{self.CAR_NAME}/sensors/core", self.speed_cb, 1)
        self.servo_sub = self.create_subscription(Float64, f"/{self.CAR_NAME}/sensors/servo_position_command", self.servo_cb, 1)

        # -------- Timer --------
        self.update_timer = self.create_timer(1.0 / self.UPDATE_RATE, self.timer_cb)

        # -------- Service --------
        self._car_reposition_srv = self.create_service(CarPose, "reposition", self._car_reposition_cb)

        # Warn throttle helper
        self._last_oob_warn_time = None

        self.get_logger().info(f"MuSHR sim node started (car_name='{self.CAR_NAME}', map_service='{self.map_service_name}')")

    """
    clip_angle: Clip an angle to be between -pi and pi
      val: Angle in radians
      Returns: Equivalent angle between -pi and pi (rad)
    """

    def clip_angle(self, val: float) -> float:
        while val > np.pi:
            val -= 2 * np.pi
        while val < -np.pi:
            val += 2 * np.pi
        return val

    """
    init_pose_cb: Callback to capture the initial pose of the car
      msg: geometry_msg/PoseStamped containing the initial pose
    """

    def init_pose_cb(self, msg: PoseStamped):
        rx_trans = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float64)
        rx_rot = utils.quaternion_to_angle(msg.pose.orientation)

        if self.map_info is not None:
            map_rx_pose = utils.world_to_map((rx_trans[0], rx_trans[1], rx_rot), self.map_info)
            if not self._check_position_in_bounds(map_rx_pose[0], map_rx_pose[1]):
                self.get_logger().warn("Requested reposition into obstacle. Ignoring.")
                return

        with self.cur_odom_to_base_lock:
            self.cur_odom_to_base_trans = rx_trans
            self.cur_odom_to_base_rot = rx_rot

    def _check_position_in_bounds(self, x, y) -> bool:
        if self.permissible_region is None:
            return True
        return (
            0 <= x < self.permissible_region.shape[1]
            and 0 <= y < self.permissible_region.shape[0]
            and bool(self.permissible_region[int(y + 0.5), int(x + 0.5)])
        )

    def speed_cb(self, msg: VescStateStamped):
        with self.last_speed_lock:
            self.last_speed = (msg.state.speed - self.SPEED_TO_ERPM_OFFSET) / self.SPEED_TO_ERPM_GAIN

    def servo_cb(self, msg: Float64):
        with self.last_steering_angle_lock:
            self.last_steering_angle = (msg.data - self.STEERING_TO_SERVO_OFFSET) / self.STEERING_TO_SERVO_GAIN

    def _warn_throttle(self, period_sec: float, text: str):
        now = self.get_clock().now()
        if self._last_oob_warn_time is None:
            self._last_oob_warn_time = now
            self.get_logger().warn(text)
            return
        if (now - self._last_oob_warn_time) > Duration(seconds=period_sec):
            self._last_oob_warn_time = now
            self.get_logger().warn(text)

    def timer_cb(self):
        now = self.get_clock().now()

        if self.last_stamp is None:
            self.last_stamp = now
        dt = (now - self.last_stamp).nanoseconds * 1e-9

        with self.last_speed_lock:
            v = self.last_speed + np.random.normal(loc=self.SPEED_OFFSET * self.last_speed, scale=self.SPEED_NOISE, size=None)

        with self.last_steering_angle_lock:
            delta = self.last_steering_angle + np.random.normal(
                loc=self.STEERING_ANGLE_OFFSET * self.last_steering_angle,
                scale=self.STEERING_ANGLE_NOISE,
                size=None,
            )

        with self.cur_odom_to_base_lock:
            new_pose = np.array(
                [self.cur_odom_to_base_trans[0], self.cur_odom_to_base_trans[1], self.cur_odom_to_base_rot],
                dtype=np.float64,
            )

            if np.abs(delta) < 1e-2:
                dtheta = 0.0
                dx = v * np.cos(self.cur_odom_to_base_rot) * dt
                dy = v * np.sin(self.cur_odom_to_base_rot) * dt

                joint_left_throttle = v * dt / self.CAR_WHEEL_RADIUS
                joint_right_throttle = v * dt / self.CAR_WHEEL_RADIUS
                joint_left_steer = 0.0
                joint_right_steer = 0.0
            else:
                tan_delta = np.tan(delta)
                dtheta = ((v / self.CAR_LENGTH) * tan_delta) * dt
                dx = (self.CAR_LENGTH / tan_delta) * (np.sin(self.cur_odom_to_base_rot + dtheta) - np.sin(self.cur_odom_to_base_rot))
                dy = (self.CAR_LENGTH / tan_delta) * (-1.0 * np.cos(self.cur_odom_to_base_rot + dtheta) + np.cos(self.cur_odom_to_base_rot))

                h_val = (self.CAR_LENGTH / tan_delta) - (self.CAR_WIDTH / 2.0)
                joint_outer_throttle = (((self.CAR_WIDTH + h_val) / (0.5 * self.CAR_WIDTH + h_val)) * v * dt) / self.CAR_WHEEL_RADIUS
                joint_inner_throttle = (((h_val) / (0.5 * self.CAR_WIDTH + h_val)) * v * dt) / self.CAR_WHEEL_RADIUS
                joint_outer_steer = np.arctan2(self.CAR_LENGTH, self.CAR_WIDTH + h_val)
                joint_inner_steer = np.arctan2(self.CAR_LENGTH, h_val)

                if delta > 0.0:
                    joint_left_throttle = joint_inner_throttle
                    joint_right_throttle = joint_outer_throttle
                    joint_left_steer = joint_inner_steer
                    joint_right_steer = joint_outer_steer
                else:
                    joint_left_throttle = joint_outer_throttle
                    joint_right_throttle = joint_inner_throttle
                    joint_left_steer = joint_outer_steer - np.pi
                    joint_right_steer = joint_inner_steer - np.pi

            # Apply noise
            new_pose[0] += (
                dx + np.random.normal(self.FORWARD_OFFSET, self.FORWARD_FIX_NOISE, None) + np.random.normal(0.0, np.abs(v) * self.FORWARD_SCALE_NOISE, None)
            )

            new_pose[1] += dy + np.random.normal(self.SIDE_OFFSET, self.SIDE_FIX_NOISE, None) + np.random.normal(0.0, np.abs(v) * self.SIDE_SCALE_NOISE, None)
            new_pose[2] += dtheta + np.random.normal(self.THETA_OFFSET, self.THETA_FIX_NOISE, None)
            new_pose[2] = self.clip_angle(new_pose[2])

            # Bounds check
            in_bounds = True
            if self.permissible_region is not None and self.map_info is not None:
                new_map_pose = np.zeros(3, dtype=np.float64)
                new_map_pose[0] = self.cur_map_to_odom_trans[0] + (
                    new_pose[0] * np.cos(self.cur_map_to_odom_rot) - new_pose[1] * np.sin(self.cur_map_to_odom_rot)
                )
                new_map_pose[1] = self.cur_map_to_odom_trans[1] + (
                    new_pose[0] * np.sin(self.cur_map_to_odom_rot) + new_pose[1] * np.cos(self.cur_map_to_odom_rot)
                )
                new_map_pose[2] = self.cur_map_to_odom_rot + new_pose[2]
                new_map_pose = utils.world_to_map(new_map_pose, self.map_info)
                in_bounds = self._check_position_in_bounds(new_map_pose[0], new_map_pose[1])
                # print(in_bounds)

            if in_bounds:
                self.cur_odom_to_base_trans[0] = new_pose[0]
                self.cur_odom_to_base_trans[1] = new_pose[1]
                self.cur_odom_to_base_rot = new_pose[2]

                self.joint_msg.position[0] = self.clip_angle(self.joint_msg.position[0] + joint_left_throttle)
                self.joint_msg.position[1] = self.clip_angle(self.joint_msg.position[1] + joint_right_throttle)
                self.joint_msg.position[2] = self.clip_angle(self.joint_msg.position[2] + joint_left_throttle)
                self.joint_msg.position[3] = self.clip_angle(self.joint_msg.position[3] + joint_right_throttle)
                self.joint_msg.position[4] = joint_left_steer
                self.joint_msg.position[5] = joint_right_steer
            else:
                self._warn_throttle(1.0, "Not in bounds")

            # TF message (utils.make_transform_msg must return TransformStamped)
            t = utils.make_transform_msg(
                self,
                self.cur_odom_to_base_trans,
                self.cur_odom_to_base_rot,
                # self.TF_PREFIX + "ground_truth_base_footprint",
                # self.TF_PREFIX + "map",
                # self.TF_PREFIX + "base_footprint",
                # self.TF_PREFIX + "map",
                self.TF_PREFIX + "base_footprint",  # child
                self.TF_PREFIX + "odom",  # parent
            )
            self.fake_laser.transform = t.transform
            self.br.sendTransform(t)

            # Joint states
            self.joint_msg.header.stamp = now.to_msg()
            self.cur_joints_pub.publish(self.joint_msg)

            self.last_stamp = now

            # PoseStamped (car pose)
            if not self.USE_MOCAP:
                cur_pose = PoseStamped()
                cur_pose.header.frame_id = "map"
                cur_pose.header.stamp = now.to_msg()
                cur_pose.pose.position.x = self.cur_odom_to_base_trans[0] + self.cur_map_to_odom_trans[0]
                cur_pose.pose.position.y = self.cur_odom_to_base_trans[1] + self.cur_map_to_odom_trans[1]
                cur_pose.pose.position.z = 0.0
                rot = self.cur_odom_to_base_rot + self.cur_map_to_odom_rot
                cur_pose.pose.orientation = utils.angle_to_quaternion(rot)
                self.state_pub.publish(cur_pose)

            # Odometry
            odom_msg = Odometry()
            odom_msg.header.stamp = self.last_stamp.to_msg()
            odom_msg.header.frame_id = self.TF_PREFIX + "odom"
            odom_msg.pose.pose.position.x = t.transform.translation.x
            odom_msg.pose.pose.position.y = t.transform.translation.y
            odom_msg.pose.pose.position.z = t.transform.translation.z
            odom_msg.pose.pose.orientation = t.transform.rotation
            odom_msg.child_frame_id = self.TF_PREFIX + "base_link"
            odom_msg.twist.twist.linear.x = dx
            odom_msg.twist.twist.linear.y = dy
            odom_msg.twist.twist.angular.z = dtheta
            self.odom_pub.publish(odom_msg)

    def _car_reposition_cb(self, request, response):
        rx_trans = np.array([request.x, request.y], dtype=np.float64)
        rx_rot = request.theta

        if self.map_info is not None:
            map_rx_pose = utils.world_to_map((rx_trans[0], rx_trans[1], rx_rot), self.map_info)
            if not self._check_position_in_bounds(map_rx_pose[0], map_rx_pose[1]):
                self.get_logger().warn("Requested reposition into obstacle. Ignoring.")
                response.success = False
                return response

        with self.cur_odom_to_base_lock:
            self.cur_odom_to_base_trans = rx_trans
            self.cur_odom_to_base_rot = rx_rot

        response.success = True
        return response

    def get_map(self):
        req = GetMap.Request()
        future = self.map_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError(f"Map service call failed: {future.exception()}")

        map_msg = future.result().map
        map_info = map_msg.info

        array_255 = np.array(map_msg.data).reshape((map_info.height, map_info.width))
        permissible_region = np.zeros_like(array_255, dtype=bool)
        permissible_region[array_255 == 0] = True

        return permissible_region, map_info, map_msg


def main(args=None):
    rclpy.init(args=args)

    node = MushrSim()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
