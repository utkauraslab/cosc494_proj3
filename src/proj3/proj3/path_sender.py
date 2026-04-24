#!/usr/bin/env python3

import argparse

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header

from mushr_interfaces.srv import CarPose
from control_interfaces.srv import FollowPath

from proj3 import utils
from proj3.path_generator import line, wave, circle, left_turn, right_turn, saw


path_generators = {
    "line": line,
    "circle": circle,
    "left-turn": left_turn,
    "right-turn": right_turn,
    "wave": wave,
    "saw": saw,
}


class PathSender(Node):
  def __init__(self):
    super().__init__("path_sender")

  def send_path(self, args):
    # Determine the args to pass through
    extra_args = vars(args).copy()
    del extra_args["speed"]
    del extra_args["reset"]
    del extra_args["path_name"]
    del extra_args["tf_prefix"]
    extra_args = {key: value for key, value in extra_args.items() if value is not None}

    start_config = (0, 0, 0)
    waypoints = path_generators[args.path_name](**extra_args)

    if args.reset:
      # reset car pose
      self.get_logger().info("Waiting for /mushr_sim/reposition service...")
      reposition_client = self.create_client(CarPose, "/mushr_sim/reposition")
      reposition_client.wait_for_service()

      req = CarPose.Request()
      req.car_name = "utk_car"
      req.x = float(start_config[0])
      req.y = float(start_config[1])
      req.theta = float(start_config[2])

      future = reposition_client.call_async(req)
      rclpy.spin_until_future_complete(self, future)

      # HACK(nickswalker5-12-21): Sim doesn't wait for reposition to propagate
      # before returning from service
      self.get_clock().sleep_for(rclpy.duration.Duration(seconds=1.0))

    h = Header()
    # Our generated paths start at (0, 0, 0), so let's make the frame correspond
    # to the current base_footprint, aka the frame where the current position is
    # (0, 0, 0)
    h.frame_id = args.tf_prefix + "base_footprint"
    h.stamp = self.get_clock().now().to_msg()

    desired_speed = args.speed

    as_poses = [
        PoseStamped(header=h, pose=utils.particle_to_pose(pose))
        for pose in waypoints
    ]
    path = Path(header=h, poses=as_poses)

    self.get_logger().info("Sending path and waiting for execution to finish")

    controller = self.create_client(FollowPath, "/control_node/follow_path")
    controller.wait_for_service()

    req = FollowPath.Request()
    req.path = path
    req.speed = float(desired_speed)


    # This call will block until the controller stops
    future = controller.call_async(req)
    rclpy.spin_until_future_complete(self, future)

    result = future.result()
    print(result)


def main(args=None):
  rclpy.init(args=args)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "path_name", type=str, choices=path_generators, help="Name of path to generate"
  )
  parser.add_argument(
      "--tf_prefix", type=str, default="", help="TF Prefix"
  )
  parser.add_argument(
      "--speed", type=float, default=1.0, help="Max speed along the path"
  )
  parser.add_argument(
      "--reset",
      action="store_true",
      default=False,
      help="Whether to reset the car position before starting",
  )
  parser.add_argument("--length", type=float, required=False, help="Length of line")
  parser.add_argument("--waypoint-sep", type=float, required=False, help="Distance between states")
  parser.add_argument("--radius", type=float, required=False, help="Radius of circle to generate")
  parser.add_argument("--turn-radius", type=float, required=False, help="Radius of turn")
  parser.add_argument("--amplitude", type=float, required=False, help="Size of signal")
  parser.add_argument("--n", type=int, required=False, help="Number of cycles")

  parsed_args, _ = parser.parse_known_args()

  node = PathSender()
  try:
    node.send_path(parsed_args)
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()
