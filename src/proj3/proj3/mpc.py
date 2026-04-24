from __future__ import division
import numpy as np

from proj3 import utils
from proj2.motion_model import KinematicCarMotionModel
from proj3.base_controller import BaseController


class ModelPredictiveController(BaseController):
  def __init__(
      self,
      car_length=0.33,
      car_width=0.15,
      collision_w=1e5,
      error_w=1.0,
      min_delta=-0.34,
      max_delta=0.34,
      K=10,
      T=5,
      kinematics_params=None,
      permissible_region=None,
      map_info=None,
      **kwargs,
  ):
    self.car_length = float(car_length)
    self.car_width = float(car_width)
    self.collision_w = float(collision_w)
    self.error_w = float(error_w)
    self.min_delta = float(min_delta)
    self.max_delta = float(max_delta)
    self.K = int(K)
    self.T = int(T)
    self.kinematics_params = kinematics_params or {}
    self.permissible_region = permissible_region
    self.map_info = map_info

    assert self.K > 0 and self.T >= 0

    self.motion_model = KinematicCarMotionModel(
        self.car_length, **self.kinematics_params
    )

    super(ModelPredictiveController, self).__init__(**kwargs)

  def sample_controls(self):
    """Sample K T-length sequences of controls (velocity, steering angle).

    In your implementation, each of the K sequences corresponds to a
    particular steering angle, applied T times. The K sequences should
    evenly span the steering angle range [self.min_delta, self.max_delta].
    """
    controls = np.empty((self.K, self.T, 2))
    controls[:, :, 0] = 0  # to be pulled from the reference path later

    # BEGIN QUESTION 4.1



    # END QUESTION 4.1

    return controls

  def get_rollout(self, pose, controls, dt=0.1):
    """For each of the K control sequences, collect the corresponding
    (T+1)-length sequence of resulting states.
    """
    assert controls.shape == (self.K, self.T, 2)

    rollouts = np.empty((self.K, self.T + 1, 3))
    rollouts[:, 0, :] = pose

    # BEGIN QUESTION 4.2



    # END QUESTION 4.2

    return rollouts

  def compute_distance_cost(self, rollouts, reference_xyt):
    """Compute the distance cost for each of the K rollouts."""
    assert rollouts.shape == (self.K, self.T + 1, 3)

    # BEGIN QUESTION 4.3



    # END QUESTION 4.3

  def compute_collision_cost(self, rollouts, _):
    """Compute the cumulative collision cost for each of the K rollouts."""
    assert rollouts.shape == (self.K, self.T + 1, 3)

    # BEGIN QUESTION 4.3



    # END QUESTION 4.3

  def compute_rollout_cost(self, rollouts, reference_xyt):
    """Compute the cumulative cost for each of the K rollouts."""
    assert rollouts.shape == (self.K, self.T + 1, 3)
    dist_cost = self.compute_distance_cost(rollouts, reference_xyt)
    coll_cost = self.compute_collision_cost(rollouts, reference_xyt)
    return dist_cost + coll_cost

  def get_error(self, pose, reference_xytv):
    # MPC uses a more complex cost function than this error vector,
    # but we need to measure the distance between the two states.
    return reference_xytv[:2] - pose[:2]

  def get_control(self, pose, reference_xytv, _):
    """Compute the MPC control law."""
    assert reference_xytv.shape[0] == 4

    # Set the velocity from the reference velocity
    self.sampled_controls[:, :, 0] = reference_xytv[3]

    # BEGIN QUESTION 4.4




    # END QUESTION 4.4

    # Set the controller's rollouts and costs (for visualization purposes).
    with self.state_lock:
      self.rollouts = rollouts
      self.costs = costs

    # BEGIN QUESTION 4.4




    # END QUESTION 4.4

  def reset_state(self):
    super(ModelPredictiveController, self).reset_state()
    with self.state_lock:
      self.sampled_controls = self.sample_controls()
      self.map_poses = np.zeros((self.K * (self.T + 1), 3))
      self.bbox_map = np.zeros((self.K * (self.T + 1), 2, 4))
      self.collisions = np.zeros(self.K * (self.T + 1), dtype=bool)
      self.obstacle_map = ~self.permissible_region
      car_l = self.car_length
      car_w = self.car_width
      self.car_bbox = (
          np.array(
              [
                  [car_l / 2.0, car_w / 2.0],
                  [car_l / 2.0, -car_w / 2.0],
                  [-car_l / 2.0, car_w / 2.0],
                  [-car_l / 2.0, -car_w / 2.0],
              ]
          )
          / self.map_info.resolution
      )

  ################################
  # Collision checking utilities #
  ################################

  def check_collisions_in_map(self, poses):
    """Compute which poses are in collision with pixels in the map."""
    assert poses.ndim == 2 and poses.shape[1] == 3

    utils.world_to_map(poses, self.map_info, out=self.map_poses)
    points = self.map_poses[:, :2]
    thetas = self.map_poses[:, 2]

    rot = np.array(
        [[np.cos(thetas), -np.sin(thetas)], [np.sin(thetas), np.cos(thetas)]]
    ).T

    self.bbox_map = (
        np.matmul(self.car_bbox[np.newaxis, ...], rot) + points[:, np.newaxis]
    )

    bbox_idx = self.bbox_map.astype(int)

    np.clip(
        bbox_idx[..., 1], 0, self.obstacle_map.shape[0] - 1, out=bbox_idx[..., 1]
    )
    np.clip(
        bbox_idx[..., 0], 0, self.obstacle_map.shape[1] - 1, out=bbox_idx[..., 0]
    )

    self.collisions = self.obstacle_map[bbox_idx[..., 1], bbox_idx[..., 0]].max(
        axis=1
    )
    return self.collisions
