#!/usr/bin/env python3
from __future__ import division

import numpy as np
import threading

import rclpy
from rclpy.node import Node


def compute_position_in_frame(p, frame):
  """Compute the position in a new coordinate frame.

  Args:
      p: vehicle state [x, y, heading]
      frame: vehicle state [x, y, heading]

  Returns:
      error: p expressed in the new coordinate frame [e_x, e_y]
  """
  # BEGIN QUESTION 1.2





  # END QUESTION 1.2


class BaseController(Node):
  def __init__(
      self,
      node_name="controller",
      frequency=10.0,
      finish_threshold=0.1,
      exceed_threshold=1.0,
      distance_lookahead=0.5,
      min_speed=0.1,
      car_length=0.33,
      tf_prefix="",
      **kwargs,
  ):
    super().__init__(node_name)

    self.frequency = float(frequency)
    self.finish_threshold = float(finish_threshold)
    self.exceed_threshold = float(exceed_threshold)
    self.distance_lookahead = float(distance_lookahead)
    self.min_speed = float(min_speed)
    self.car_length = float(car_length)
    self.tf_prefix = str(tf_prefix)

    # Error is typically going to be near this distance lookahead parameter,
    # so if the exceed threshold is lower we'll just immediately error out
    assert self.distance_lookahead < self.exceed_threshold

    self.path = None
    self.ready_event = threading.Event()
    self.path_condition = threading.Condition()
    self.finished_event = threading.Event()
    self.looped_event = threading.Event()
    self.shutdown_event = threading.Event()
    self.state_lock = threading.RLock()
    self.reset_state()

    self.current_pose = None

    self._timer = None
    self._interval = 1.0 / self.frequency

  def get_reference_index(self, pose, path_xytv, distance_lookahead):
    """Return the index to the next control target on the reference path.

    To compute a reference state that is some lookahead distance away, we
    recommend first finding the state on the path that is closest to the
    current vehicle state. Then, step along the path's waypoints and select
    the index of the first state that is greater than the lookahead distance
    from the current state. (You can also look back one index to see which
    state is closer to the desired lookahead distance.)

    Note that this method must be computationally efficient, since it runs
    directly in the control loop. Vectorize where you can.

    Args:
        pose: current state of the vehicle [x, y, heading]
        path_xytv: np.array of states and speeds with shape L x 4
        distance_lookahead (float): lookahead distance

    Returns:
        index to the reference state on the path
    """
    with self.path_condition:
      # Hint: compute all the distances from the current state to the
      # path's waypoints. You may find the `argmin` method useful.

      # BEGIN QUESTION 1.1





      # END QUESTION 1.1

      return len(path_xytv) - 1

  def get_error(self, pose, reference_xytv):
    """Compute the error vector."""
    raise NotImplementedError

  def get_control(self, pose, reference_xytv, error):
    """Compute the control action."""
    raise NotImplementedError

  def path_complete(self, pose, error, distance_lookahead):
    """Return whether the reference path has been completed."""
    ref_is_path_end = self.get_reference_index(
        pose, self.path, distance_lookahead
    ) == (len(self.path) - 1)
    err_l2 = np.linalg.norm(error)
    within_error = err_l2 < self.finish_threshold
    beyond_exceed = err_l2 > self.exceed_threshold
    return ref_is_path_end and within_error, beyond_exceed

  ################
  # Control Loop #
  ################

  def start(self):
    """Start the control loop."""
    if self._timer is None:
      self._timer = self.create_timer(self._interval, self._control_loop)
    self.ready_event.set()

  def _control_loop(self):
    """Implement the control loop."""
    if self.shutdown_event.is_set():
      return

    with self.path_condition:
      if self.path is None:
        return

      if len(self.path) == 0:
        self.completed = True
        self.errored = False
        self.path = None
        self.finished_event.set()
        self.finished_event.clear()
        return

      with self.state_lock:
        if self.current_pose is None:
          return

        index = self.get_reference_index(
            self.current_pose, self.path, self.distance_lookahead
        )
        self.selected_pose = self.get_reference_pose(index)

        # ROS2 logging
        # self.get_logger().info(
        #     f"index={index}, current_pose={self.current_pose}, selected_pose={self.selected_pose}"
        # )

        self.error = np.linalg.norm(
            self.selected_pose[:2] - self.current_pose[:2]
        )

        error = self.get_error(self.current_pose, self.selected_pose)
        self.next_ctrl = self.get_control(
            self.current_pose, self.selected_pose, error
        )

        complete, errored = self.path_complete(
            self.current_pose, error, self.distance_lookahead
        )

        if self.prev_pose is None:
          self.prev_pose = self.current_pose
          self.prev_pose_stamp = self.get_clock().now()
        else:
          progress = np.linalg.norm(
              np.array(self.prev_pose[:2]) - self.current_pose[:2]
          )
          if progress > 0.5:
            self.prev_pose = self.current_pose
            self.prev_pose_stamp = self.get_clock().now()
          else:
            dt = (
                self.get_clock().now() - self.prev_pose_stamp
            ).nanoseconds / 1e9
            if dt > 10.0:
              errored = True

        self.looped_event.set()
        self.looped_event.clear()

        # Clear this out so we don't repeat the exact same calculations.
        self.current_pose = None

        if complete or errored:
          self.errored = errored
          self.completed = complete

          # Send one final zero command so the vehicle stops
          self.next_ctrl = np.array([0.0, 0.0])

          # Wake result listener so it publishes the stop command
          self.looped_event.set()
          self.looped_event.clear()

          self.path = None
          self.finished_event.set()
          self.finished_event.clear()

  def shutdown(self):
    """Shut down the controller."""
    self.shutdown_event.set()

    if self._timer is not None:
      self._timer.cancel()
      self.destroy_timer(self._timer)
      self._timer = None

    with self.path_condition:
      self.path_condition.notify_all()

    self.looped_event.set()
    self.get_logger().info("Control loop ending")

  def reset_state(self):
    """Reset the controller's internal state."""
    with self.state_lock:
      self.selected_pose = None
      self.error = None
      self.next_ctrl = None
      self.rollouts = None
      self.costs = None
      self.errored = None
      self.completed = None
      self.prev_pose = None
      self.prev_pose_stamp = None

  ####################
  # Helper Functions #
  ####################

  def is_alive(self):
    """Return whether controller is ready to begin tracking."""
    return self.ready_event.is_set()

  def get_reference_pose(self, index):
    """Return the reference state from the reference path at the index."""
    with self.path_condition:
      assert len(self.path) > index
      return self.path[index]

  def set_path(self, path):
    """Set the reference path to track.

    This implicitly resets the internal state of the controller.
    """
    with self.path_condition:
      self.path = path
      self.reset_state()
      self.path_condition.notify_all()

  def cancel_path(self):
    """Cancel the current path being tracked, if one exists."""
    if self.path is None:
      return False

    with self.path_condition:
      self.path = None
      self.path_condition.notify_all()

    with self.state_lock:
      self.completed = False
      self.errored = False
      self.finished_event.set()
      self.finished_event.clear()

    self.looped_event.set()
    self.looped_event.clear()
    return True


def time_parameterize_ramp_up_ramp_down(path_xyt, speed, min_speed):
  """Parameterize a geometric path of states with a desired speed.

  Vehicles can't instantly reach the desired speed, so we need to ramp up to
  full speed and ramp down to 0 at the end of the path.

  Args:
      path_xyt: np.array of states with shape L x 3
      speed (double): desired speed

  Returns:
      path_xytv: np.array of states and speed with shape L x 4
  """
  if path_xyt.shape[0] < 4:
    speeds = speed * np.ones(path_xyt.shape[0])
    speeds = np.maximum(speeds, min_speed)
    return np.hstack([path_xyt, speeds[:, np.newaxis]])

  ramp_distance = 0.5
  displacement_vectors = np.diff(path_xyt[:, [0, 1]], axis=0)
  displacements = np.linalg.norm(displacement_vectors, axis=1)
  cumulative_path_length = np.cumsum(displacements)
  ramp_mask = (cumulative_path_length < ramp_distance) | (
      displacements.sum() - ramp_distance < cumulative_path_length
  )
  ramp_mask[[0, -1]] = True
  change_points = np.where(np.diff(ramp_mask))[0]
  speeds = np.interp(
      np.arange(len(path_xyt)) / len(path_xyt),
      [0.0, change_points[0] / len(path_xyt), change_points[1] / len(path_xyt), 1.0],
      [0, speed, speed, 0],
  )
  speeds = np.maximum(speeds, min_speed)
  return np.hstack([path_xyt, speeds[:, np.newaxis]])
