from __future__ import division

import threading

import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import rclpy
import tf2_ros
import tf2_geometry_msgs

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Float64, Header
from visualization_msgs.msg import Marker, MarkerArray

from proj3.base_controller import time_parameterize_ramp_up_ramp_down
from proj3.pid import PIDController
from proj3.purepursuit import PurePursuitController
from proj3.mpc import ModelPredictiveController
from control_interfaces.srv import FollowPath
from proj3 import utils


controllers = {
    "pid": PIDController,
    "pp": PurePursuitController,
    "mpc": ModelPredictiveController,
}


class ControlROS(Node):
  def __init__(self, controller, tf_prefix="", transform_listener=None, node_name="control_ros"):
    super().__init__(node_name)

    self.controller = controller
    self.tf_prefix = tf_prefix

    if transform_listener is not None:
      self.tf_buffer = transform_listener.buffer
      self.tl = transform_listener
    else:
      self.tf_buffer = tf2_ros.Buffer()
      self.tl = tf2_ros.TransformListener(self.tf_buffer, self)

    self._pose_thread = None
    self._result_thread = None
    self._real_poses = []

  def start(self):
    self.setup_pub_sub()
    self._pose_thread = threading.Thread(target=self.__pose_updater, daemon=True)
    self._pose_thread.start()
    self._result_thread = threading.Thread(target=self.__result_listener, daemon=True)
    self._result_thread.start()
    self.controller.start()

    import time
    time.sleep(0.5)

  def shutdown(self):
    self.controller.shutdown()

    if hasattr(self, "_follow_path_srv"):
      self.destroy_service(self._follow_path_srv)
    if hasattr(self, "_control_pub"):
      self.destroy_publisher(self._control_pub)
    if hasattr(self, "_error_pub"):
      self.destroy_publisher(self._error_pub)
    if hasattr(self, "_reference_state_pub"):
      self.destroy_publisher(self._reference_state_pub)
    if hasattr(self, "_path_pub"):
      self.destroy_publisher(self._path_pub)
    if hasattr(self, "_real_path_pub"):
      self.destroy_publisher(self._real_path_pub)

  def __pose_updater(self):
    import time

    pose_period = 0.01
    while rclpy.ok() and not self.controller.shutdown_event.is_set():
      if self.controller.path is None:
        self._real_poses = []
        with self.controller.path_condition:
          while (
              self.controller.path is None
              and not self.controller.shutdown_event.is_set()
          ):
            self.controller.path_condition.wait(timeout=0.1)
        if self.controller.shutdown_event.is_set():
          break

      latest_pose = self._get_car_pose()
      if latest_pose is None:
        time.sleep(pose_period)
        continue

      with self.controller.state_lock:
        self.controller.current_pose = latest_pose

      self._real_poses.append(utils.particle_to_pose(latest_pose))
      time.sleep(pose_period)

    self.get_logger().info("Pose update shutdown")

  def __result_listener(self):
    i = 0
    while rclpy.ok() and not self.controller.shutdown_event.is_set():
      i += 1
      self.controller.looped_event.wait(timeout=0.1)

      if self.controller.shutdown_event.is_set():
        continue

      if self.controller.selected_pose is not None:
        pose = self.controller.selected_pose
        p = PoseStamped()
        p.header = Header()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = "map"
        p.pose.position.x = pose[0]
        p.pose.position.y = pose[1]
        p.pose.orientation = utils.angle_to_quaternion(pose[2])
        self._reference_state_pub.publish(p)

      if self.controller.error is not None:
        self._error_pub.publish(Float64(data=float(self.controller.error)))

      if self.controller.next_ctrl is not None:
        ctrl = self.controller.next_ctrl
        assert len(ctrl) == 2
        ctrlmsg = AckermannDriveStamped()
        ctrlmsg.header.stamp = self.get_clock().now().to_msg()
        ctrlmsg.drive.speed = float(ctrl[0])
        ctrlmsg.drive.steering_angle = float(ctrl[1])
        self._control_pub.publish(ctrlmsg)

      if len(self._real_poses) > 0 and i % 10 == 0:
        path = PoseArray()
        path.header = Header()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = self._real_poses
        self._real_path_pub.publish(path)

    self.get_logger().info("Result monitor shutdown")

  def setup_pub_sub(self):
    """Initialize ROS service and publishers."""
    self._follow_path_srv = self.create_service(
        FollowPath, "~/follow_path", self.cb_path
    )

    self._control_pub = self.create_publisher(
        AckermannDriveStamped,
        "path_control",
        2,
    )

    self._error_pub = self.create_publisher(
        Float64,
        "~/error",
        1,
    )

    self._reference_state_pub = self.create_publisher(
        PoseStamped,
        "~/path/reference_state",
        1,
    )

    self._path_pub = self.create_publisher(
        PoseArray, "~/path/poses", 1
    )

    self._real_poses = []
    self._real_path_pub = self.create_publisher(
        PoseArray, "~/real_path/poses", 1
    )

  def reset_state(self):
    """Reset the controller's internal state (e.g., accumulators in PID control)."""
    self.get_logger().info("Start state reset")
    self.controller.reset_state()
    self.get_logger().info("End state reset")
    return []

  def follow_path_with_speed(self, path_xyt, speed):
    """Follow a geometric path of states with a desired speed."""
    if not self.controller.is_alive():
      raise RuntimeError("Path command set before controller was started")
    path_xytv = time_parameterize_ramp_up_ramp_down(
        path_xyt, speed, self.controller.min_speed
    )
    self.follow_path(path_xytv)

  def follow_path(self, path_xytv):
    """Follow a geometric path of states and pre-specified speed."""
    if not self.controller.is_alive():
      raise RuntimeError("Path command set before controller was started")
    path_pose_array = configs_to_pose_array(path_xytv[:, :3], node=self)
    self._path_pub.publish(path_pose_array)
    self.controller.set_path(path_xytv)
    self.get_logger().info("Path set")

  def cb_path(self, request, response):
    """Handle a new geometric path tracking request."""
    self.controller.cancel_path()
    speed = request.speed

    transformed_path = []
    for pose in request.path.poses:
      transformed_pose = self.tf_buffer.transform(
          pose,
          "map",
          timeout=Duration(seconds=1.0),
      )
      transformed_path.append(transformed_pose)

    path_xyt = np.array(
        [utils.pose_to_particle(pose.pose) for pose in transformed_path]
    )
    self.follow_path_with_speed(path_xyt, speed)
    finished, error = self.wait_for_finish()

    response.finished = finished
    response.error = error
    return response

  def wait_for_finish(self, timeout=None):
    """Wait for the controller to terminate efforts on the current path."""
    if self.controller.path is not None:
      self.controller.finished_event.wait(timeout=timeout)
    return bool(self.controller.completed), bool(self.controller.errored)

  def _get_car_pose(self):
    """Return the current vehicle state."""
    try:
      transform = self.tf_buffer.lookup_transform(
          "map",
          self.tf_prefix + "base_footprint",
          rclpy.time.Time(),
      )
      transform = transform.transform
      return np.array(
          [
              transform.translation.x,
              transform.translation.y,
              utils.quaternion_to_angle(transform.rotation),
          ]
      )
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
      self.get_logger().warning(str(e))
      return None


def declare_if_needed(node, name, default):
  if not node.has_parameter(name):
    node.declare_parameter(name, default)


def get_ros_params(node):
  """Pull controller parameters from ROS2 parameters."""

  defaults = [
      ("car_length", 0.33),
      ("frequency", 50.0),
      ("finish_threshold", 0.3),
      ("exceed_threshold", 4.0),
      ("distance_lookahead", 0.6),
      ("min_speed", 0.5),
      ("type", "pid"),

      ("pid.kp", 0.5),
      ("pid.kd", 0.5),

      ("pp.frequency", 50.0),
      ("pp.finish_threshold", 0.3),
      ("pp.exceed_threshold", 4.0),
      ("pp.distance_lookahead", 0.6),
      ("pp.min_speed", 0.5),
      ("pp.car_length", 0.33),

      ("mpc.car_width", 0.15),
      ("mpc.collision_w", 1e5),
      ("mpc.error_w", 1.0),
      ("mpc.min_delta", -0.34),
      ("mpc.max_delta", 0.34),
      ("mpc.K", 2),
      ("mpc.T", 1),
      ("mpc.frequency", 50.0),
      ("mpc.finish_threshold", 0.3),
      ("mpc.exceed_threshold", 4.0),
      ("mpc.distance_lookahead", 0.6),
      ("mpc.min_speed", 0.5),
      ("mpc.car_length", 0.33),

      ("motion_params.vel_std", 0.0),
      ("motion_params.delta_std", 0.0),
      ("motion_params.x_std", 0.0),
      ("motion_params.y_std", 0.0),
      ("motion_params.theta_std", 0.0),
  ]

  for name, default in defaults:
    declare_if_needed(node, name, default)

  base_params = {
      "car_length": float(node.get_parameter("car_length").value),
      "frequency": float(node.get_parameter("frequency").value),
      "finish_threshold": float(node.get_parameter("finish_threshold").value),
      "exceed_threshold": float(node.get_parameter("exceed_threshold").value),
      "distance_lookahead": float(node.get_parameter("distance_lookahead").value),
      "min_speed": float(node.get_parameter("min_speed").value),
  }

  controller_type = str(node.get_parameter("type").value).lower()

  if controller_type == "pid":
    params = {
        "kp": float(node.get_parameter("pid.kp").value),
        "kd": float(node.get_parameter("pid.kd").value),
    }

  elif controller_type == "pp":
    params = {
        "frequency": float(node.get_parameter("pp.frequency").value),
        "finish_threshold": float(node.get_parameter("pp.finish_threshold").value),
        "exceed_threshold": float(node.get_parameter("pp.exceed_threshold").value),
        "distance_lookahead": float(node.get_parameter("pp.distance_lookahead").value),
        "min_speed": float(node.get_parameter("pp.min_speed").value),
        "car_length": float(node.get_parameter("pp.car_length").value),
    }

  elif controller_type == "mpc":
    params = {
        "car_width": float(node.get_parameter("mpc.car_width").value),
        "collision_w": float(node.get_parameter("mpc.collision_w").value),
        "error_w": float(node.get_parameter("mpc.error_w").value),
        "min_delta": float(node.get_parameter("mpc.min_delta").value),
        "max_delta": float(node.get_parameter("mpc.max_delta").value),
        "K": int(node.get_parameter("mpc.K").value),
        "T": int(node.get_parameter("mpc.T").value),
        "frequency": float(node.get_parameter("mpc.frequency").value),
        "finish_threshold": float(node.get_parameter("mpc.finish_threshold").value),
        "exceed_threshold": float(node.get_parameter("mpc.exceed_threshold").value),
        "distance_lookahead": float(node.get_parameter("mpc.distance_lookahead").value),
        "min_speed": float(node.get_parameter("mpc.min_speed").value),
        "car_length": float(node.get_parameter("mpc.car_length").value),
        "kinematics_params": {
            "vel_std": float(node.get_parameter("motion_params.vel_std").value),
            "delta_std": float(node.get_parameter("motion_params.delta_std").value),
            "x_std": float(node.get_parameter("motion_params.x_std").value),
            "y_std": float(node.get_parameter("motion_params.y_std").value),
            "theta_std": float(node.get_parameter("motion_params.theta_std").value),
        },
    }

    permissible_region, map_info = utils.get_map("/static_map")
    params["permissible_region"] = permissible_region
    params["map_info"] = map_info

  else:
    raise RuntimeError(
        f"'{controller_type}' is not a controller. You must specify 'pid', 'pp', or 'mpc'"
    )

  merged_params = base_params.copy()
  merged_params.update(params)
  return controller_type, merged_params


def rollouts_to_markers_cmap(poses, costs, ns="paths", cmap="cividis", scale=0.01, node=None):
  max_c = np.max(costs)
  min_c = np.min(costs)
  norm = colors.Normalize(vmin=min_c, vmax=max_c)

  if cmap not in cm.cmaps_listed.keys():
    cmap = "viridis"
  cmap = cm.get_cmap(name=cmap)

  def colorfn(cost):
    r, g, b, a = 0.0, 0.0, 0.0, 1.0
    col = cmap(norm(cost))
    r, g, b = col[0], col[1], col[2]
    if len(col) > 3:
      a = col[3]
    return r, g, b, a

  return rollouts_to_markers(poses, costs, colorfn, ns, scale, node=node)


def rollouts_to_markers(poses, costs, colorfn, ns="paths", scale=0.01, node=None):
  assert poses.shape[0] == costs.shape[0]

  markers = MarkerArray()
  stamp = node.get_clock().now().to_msg() if node is not None else rclpy.clock.Clock().now().to_msg()

  for i, (traj, cost) in enumerate(zip(poses, costs)):
    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp = stamp
    m.ns = ns + str(i)
    m.id = i
    m.type = m.LINE_STRIP
    m.action = m.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = scale
    m.color.r, m.color.g, m.color.b, m.color.a = colorfn(cost)

    for t in traj:
      p = Point()
      p.x, p.y = t[0], t[1]
      m.points.append(p)

    markers.markers.append(m)
  return markers


def configs_to_pose_array(path_xyt, node=None):
  """Publish path visualization messages."""
  path_as_poses = list(map(utils.particle_to_pose, path_xyt))
  pa = PoseArray()
  pa.header = Header()
  pa.header.frame_id = "map"
  if node is not None:
    pa.header.stamp = node.get_clock().now().to_msg()
  pa.poses = path_as_poses
  return pa
