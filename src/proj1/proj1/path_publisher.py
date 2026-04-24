#!/usr/bin/env python3
#

import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration  # ROS2 duration

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from proj1 import utils


class PathPublisher(Node):
    def __init__(self, control_duration: float = 1.0, control_rate: float = 10.0):
        """Create a PathPublisher.

        Args:
          control_duration: How long to send each command (in seconds)
          control_rate: How frequently to send each command (in Hz)
        """
        super().__init__("path_publisher")

        self._setup_publishers()
        self.control_duration = float(control_duration)
        self.control_rate_hz = float(control_rate)

        # Parse and validate the plan file. The first line of the plan file is
        # the initial pose [x, y, theta]. All subsequent elements are Ackermann
        # commands [speed, steering_angle].
        #
        # ROS2: use a parameter (same name), declared with a default.
        self.declare_parameter("plan_file", "")
        plan_file = self.get_parameter("plan_file").get_parameter_value().string_value
        if not plan_file:
            raise RuntimeError("Parameter '~plan_file' (ROS1) / 'plan_file' (ROS2) is required but empty.")

        with open(plan_file, "r", encoding="utf-8") as f:
            plan = [list(map(float, line.split(","))) for line in f if line.strip()]

        self.get_logger().info(f"path = {plan}")

        self.init_pose = plan[0]
        self.commands = plan[1:]
        assert len(self.init_pose) == 3, len(self.init_pose)
        for cmd in self.commands:
            assert len(cmd) == 2, len(cmd)

        # ---------------------------
        # Option A: timer-driven plan execution (non-blocking, ROS2-correct)
        # ---------------------------
        self._started = False
        self._done = False

        self._cmd_idx = 0
        self._active_cmd_msg = None
        self._active_cmd_end_time = None  # rclpy.time.Time

        # Timer publishes at control_rate_hz, so we don't need rate.sleep()
        self._timer = self.create_timer(1.0 / max(self.control_rate_hz, 1e-6), self._tick)

    def follow_plan(self):
        """Follow the parsed plan.

        ROS2 note:
          - This starts the plan execution.
          - It does NOT block; you must spin the node to execute.
        """
        if self._started:
            return

        self._started = True

        # Publish initial pose once
        self.init_pose_publisher.publish(self.make_pose_msg(self.init_pose))

        # Start with the first command
        self._cmd_idx = 0
        self._start_next_command()

    def is_done(self) -> bool:
        return self._done

    def _start_next_command(self):
        """Load the next command and set its end time."""
        if self._cmd_idx >= len(self.commands):
            self._done = True
            self.get_logger().info("Plan complete.")
            return

        cmd = self.commands[self._cmd_idx]
        self._active_cmd_msg = self.make_command_msg(cmd)
        self._active_cmd_end_time = self.get_clock().now() + Duration(seconds=self.control_duration)

        self.get_logger().info(f"Publishing command {cmd} for {self.control_duration:.2f}s")

        self._cmd_idx += 1

    def _tick(self):
        """Timer callback: publish current command and advance when time is up."""
        if not self._started or self._done:
            return

        # Publish current command at fixed rate
        if self._active_cmd_msg is not None:
            self.control_publisher.publish(self._active_cmd_msg)

        # Advance to next command when duration elapsed
        if self._active_cmd_end_time is not None and self.get_clock().now() >= self._active_cmd_end_time:
            self._start_next_command()

    def make_pose_msg(self, pose_data):
        """Create a PoseWithCovarianceStamped message from an initial pose."""
        msg = PoseWithCovarianceStamped()

        # NOTE: If utils.make_header uses rospy.Time.now(), you may need to update
        # it for ROS2 (builtin_interfaces/msg/Time) or pass in self.get_clock().now().
        msg.header = utils.make_header(frame_id="map", node=self)

        msg.pose.pose = utils.particle_to_pose(list(map(float, pose_data)))
        return msg

    def make_command_msg(self, cmd):
        """Create an AckermannDriveStamped message from a command."""
        v, delta = cmd
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(delta)
        msg.drive.speed = float(v)
        return msg

    def _setup_publishers(self):
        """Set up publishers: one for the initial pose, one for each command."""
        # ROS2: create_publisher(msg_type, topic_name, qos_depth)
        self.init_pose_publisher = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)
        self.control_publisher = self.create_publisher(AckermannDriveStamped, "path_control", 1)

        # Publishers sometimes need time to warm up. You can also wait until there
        # are subscribers to start publishing. (See publisher documentation.)
        time.sleep(1.0)


def main():
    rclpy.init()
    node = PathPublisher()

    try:
        # Start the plan, then spin until it finishes.
        node.follow_plan()
        while rclpy.ok() and not node.is_done():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
