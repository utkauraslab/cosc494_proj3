from __future__ import division
import numpy as np

from proj3.base_controller import BaseController
from proj3.base_controller import compute_position_in_frame


class PurePursuitController(BaseController):
  def __init__(self, car_length=0.33, **kwargs):
    self.car_length = float(car_length)

    # Get the keyword args that we didn't consume with the above initialization
    super(PurePursuitController, self).__init__(**kwargs)


  def get_error(self, pose, reference_xytv):
    """Compute the Pure Pursuit error.

    Args:
        pose: current state of the vehicle [x, y, heading]
        reference_xytv: reference state and speed

    Returns:
        error: Pure Pursuit error
    """
    return compute_position_in_frame(reference_xytv[:3], pose)


  def get_control(self, pose, reference_xytv, error):
    """Compute the Pure Pursuit control law.

    Args:
        pose: current state of the vehicle [x, y, heading]
        reference_xytv: reference state and speed
        error: error vector from get_error

    Returns:
        control: np.array of velocity and steering angle
    """
    # BEGIN QUESTION 3.1





    # END QUESTION 3.1
