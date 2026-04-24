#!/usr/bin/env python3
# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team
# License: BSD 3-Clause. See LICENSE.md file in root directory.
#
# Modified to ROS2 by Fei Liu, University of Tennessee, 2026

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped


def _ns_join(*parts: str) -> str:
    """
    Join ROS names with exactly one slash between tokens, and ensure leading slash.
    """
    cleaned = [p.strip("/") for p in parts if p is not None and str(p) != ""]
    return "/" + "/".join(cleaned)


class FakeVescDriver:
    def __init__(self, node: Node, car_name: str):
        self.node = node
        self.car_name = (car_name or "car").strip("/")

        # Topics consistent with MushrSim subscriptions:
        #   /{car}/vesc/sensors/core
        #   /{car}/vesc/sensors/servo_position_command
        self.topic_cmd_speed = _ns_join(self.car_name, "commands", "motor", "speed")
        self.topic_cmd_servo = _ns_join(self.car_name, "commands", "servo", "position")
        self.topic_state_core = _ns_join(self.car_name, "sensors", "core")
        self.topic_servo_out = _ns_join(self.car_name, "sensors", "servo_position_command")

        # Subscriptions (commands)
        self.speed_sub = self.node.create_subscription(
            Float64,
            self.topic_cmd_speed,
            self.speed_cb,
            10,
        )

        self.servo_position_sub = self.node.create_subscription(
            Float64,
            self.topic_cmd_servo,
            self.servo_position_cb,
            10,
        )

        # Publications (sensors)
        self.state_pub = self.node.create_publisher(
            VescStateStamped,
            self.topic_state_core,
            10,
        )

        self.servo_pub = self.node.create_publisher(
            Float64,
            self.topic_servo_out,
            10,
        )

        self.node.get_logger().info(
            "FakeVescDriver topics:\n"
            f"  sub speed: {self.topic_cmd_speed}\n"
            f"  sub servo: {self.topic_cmd_servo}\n"
            f"  pub core : {self.topic_state_core}\n"
            f"  pub servo: {self.topic_servo_out}"
        )

    def speed_cb(self, msg: Float64):
        vss = VescStateStamped()
        vss.header.stamp = self.node.get_clock().now().to_msg()
        # MushrSim expects msg.state.speed (it later converts ERPM->m/s using gain/offset)
        vss.state.speed = float(msg.data)
        self.state_pub.publish(vss)

    def servo_position_cb(self, msg: Float64):
        out_msg = Float64()
        out_msg.data = float(msg.data)
        self.servo_pub.publish(out_msg)


class FakeVescDriverNode(Node):
    def __init__(self):
        super().__init__("fake_vesc_driver_node")

        # Match MushrSim parameter name: "car_name"
        self.declare_parameter("car_name", "car")
        car_name = self.get_parameter("car_name").get_parameter_value().string_value.strip() or "car"

        self.fvs = FakeVescDriver(self, car_name=car_name)


def main(args=None):
    rclpy.init(args=args)
    node = FakeVescDriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
