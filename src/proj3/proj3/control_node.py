#!/usr/bin/env python3

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .control_ros2 import ControlROS, get_ros_params, controllers


def main(args=None):
  rclpy.init(args=args)

  node = ControlROS(controller=None, node_name="control_node")

  controller_type, params = get_ros_params(node)
  node.get_logger().info(f"Using {controller_type} controller")

  controller = controllers[controller_type](**params)
  node.controller = controller

  node.start()

  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.shutdown()
    controller.destroy_node()
    node.destroy_node()
    rclpy.shutdown()
