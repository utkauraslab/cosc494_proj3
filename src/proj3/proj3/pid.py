from __future__ import division

import numpy as np

from .base_controller import BaseController
from .base_controller import compute_position_in_frame


class PIDController(BaseController):
  def __init__(self, node_name="pid_controller", kp=0.5, kd=0.5, **kwargs):
    # Initialize ROS2 node first
    self.kp = float(kp)
    self.kd = float(kd)

    super(PIDController, self).__init__(node_name=node_name, **kwargs)

  def get_error(self, pose, reference_xytv):
    """Compute the PD error.

    Args:
        pose: current state of the vehicle [x, y, heading]
        reference_xytv: reference state and speed

    Returns:
        error: across-track and cross-track error
    """
    return compute_position_in_frame(pose, reference_xytv[:3])

  def get_control(self, pose, reference_xytv, error):
    """Compute the PD control law.

    Args:
        pose: current state of the vehicle [x, y, heading]
        reference_xytv: reference state and speed
        error: error vector from get_error

    Returns:
        control: np.array of velocity and steering angle
            (velocity should be copied from reference velocity)
    """
    # BEGIN QUESTION 2.1




    # END QUESTION 2.1
