#!/usr/bin/env python3

import rclpy
from rclpy.executors import MultiThreadedExecutor

from .control_ros2 import ControlROS, get_ros_params, controllers


def main(args=None):
  rclpy.init(args=args)

  param_node = rclpy.create_node("control_param_loader")
  controller_type, params = get_ros_params(param_node)
  param_node.get_logger().info(f"Using {controller_type} controller")

  controller = controllers[controller_type](**params)
  node = ControlROS(controller)

  node.start()

  executor = MultiThreadedExecutor()
  executor.add_node(node)
  executor.add_node(controller)

  try:
    executor.spin()
  except KeyboardInterrupt:
    pass
  finally:
    node.shutdown()
    executor.remove_node(node)
    executor.remove_node(controller)

    node.destroy_node()
    controller.destroy_node()
    param_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
