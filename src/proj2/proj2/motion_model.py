#!/usr/bin/env python3
from __future__ import division

from threading import Lock
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped
import pdb


class KinematicCarMotionModel:
    """The kinematic car motion model."""

    def __init__(self, car_length, **kwargs):
        """Initialize the kinematic car motion model.

        Args:
            car_length: the length of the car
            **kwargs (object): any number of optional keyword arguments:
                vel_std (float): std dev of the control velocity noise
                delta_std (float): std dev of the control delta noise
                x_std (float): std dev of the x position noise
                y_std (float): std dev of the y position noise
                theta_std (float): std dev of the theta noise
        """
        defaults = {
            "vel_std": 0.05,
            "delta_std": 0.5,
            "x_std": 0.05,
            "y_std": 0.05,
            "theta_std": 0.05,
        }
        if not set(kwargs).issubset(set(defaults)):
            raise ValueError("Invalid keyword argument provided")

        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)

        if car_length <= 0.0:
            raise ValueError("The model is only defined for positive, non-zero car lengths")
        self.car_length = car_length

    def compute_changes(self, states, controls, dt, delta_threshold=1e-2):
        """Integrate the (deterministic) kinematic car model.

        Given vectorized states and controls, compute the changes in state when
        applying the control for duration dt.

        If the absolute value of the applied delta is below delta_threshold,
        round down to 0. We assume that the steering angle (and therefore the
        orientation component of state) does not change in this case.

        Args:
            states: np.array of states with shape M x 3
            controls: np.array of controls with shape M x 2
            dt (float): control duration
            delta_threshold (float): steering angle threshold

        Returns:
            M x 3 np.array, where the three columns are dx, dy, dtheta
        """
        # BEGIN QUESTION 1.1
        theta = states[:, 2]
        v = controls[:, 0]
        delta = controls[:, 1]

        changes = np.zeros_like(states, dtype=float)

        straight = np.abs(delta) < delta_threshold
        turning = ~straight

        changes[straight, 0] = v[straight] * np.cos(theta[straight]) * dt
        changes[straight, 1] = v[straight] * np.sin(theta[straight]) * dt
        changes[straight, 2] = 0.0

        if np.any(turning):
            theta_t = theta[turning]
            v_t = v[turning]
            delta_t = delta[turning]

            dtheta = (v_t / self.car_length) * np.tan(delta_t) * dt
            theta_next = theta_t + dtheta
            radius = self.car_length / np.tan(delta_t)

            changes[turning, 0] = radius * (np.sin(theta_next) - np.sin(theta_t))
            changes[turning, 1] = radius * (-np.cos(theta_next) + np.cos(theta_t))
            changes[turning, 2] = dtheta

        return changes
        # END QUESTION 1.1

    def apply_motion_model(self, states, vel, delta, dt):
        """Propagate states through the noisy kinematic car motion model.

        Given the nominal control (vel, delta), sample M noisy controls.
        Then, compute the changes in state with the noisy controls.
        Finally, add noise to the resulting states.

        NOTE: This function does not have a return value: your implementation
        should modify the states argument in-place with the updated states.

        >>> states = np.ones((3, 2))
        >>> states[2, :] = np.arange(2)  # modifies the row at index 2
        >>> a = np.array([[1, 2], [3, 4], [5, 6]])
        >>> states[:] = a + a            # modifies states; note the [:]

        Args:
            states: np.array of states with shape M x 3
            vel (float): nominal control velocity
            delta (float): nominal control steering angle
            dt (float): control duration
        """
        n_particles = states.shape[0]

        # Hint: you may find the np.random.normal function useful
        # BEGIN QUESTION 1.2
        # noisy_controls = np.zeros((n_particles, 2), dtype=float)
        # noisy_controls[:, 0] = np.random.normal(vel, self.vel_std, n_particles)
        # noisy_controls[:, 1] = np.random.normal(delta, self.delta_std, n_particles)

        # changes = self.compute_changes(states, noisy_controls, dt)

        # changes[:, 0] += np.random.normal(0.0, self.x_std, n_particles)
        # changes[:, 1] += np.random.normal(0.0, self.y_std, n_particles)
        # changes[:, 2] += np.random.normal(0.0, self.theta_std, n_particles)

        # states[:] = states + changes
        # states[:, 2] = (states[:, 2] + np.pi) % (2 * np.pi) - np.pi

        noisy_controls = np.zeros((n_particles, 2), dtype=float)
        noisy_controls[:, 0] = np.random.normal(vel, self.vel_std, n_particles)
        noisy_controls[:, 1] = np.random.normal(delta, self.delta_std, n_particles)

        changes = self.compute_changes(states, noisy_controls, dt)

        states[:] = states + changes
        states[:, 2] = (states[:, 2] + np.pi) % (2 * np.pi) - np.pi
        # END QUESTION 1.2


class KinematicCarMotionModelROS(Node):
    """A ROS subscriber that applies the kinematic car motion model.

    This applies the motion model to the particles whenever it receives a
    message from the control topic. Each particle represents a state (pose).

    These implementation details can be safely ignored, although you're welcome
    to continue reading to better understand how the entire state estimation
    pipeline is connected.
    """

    def __init__(self, particles, noise_params=None, state_lock=None, **kwargs):
        """Initialize the kinematic car model ROS subscriber.

        Args:
            particles: the particles to update in-place
            noise_params: a dictionary of parameters for the motion model
            state_lock: guarding access to the particles during update,
                since it is shared with other processes
            **kwargs: must include
                motor_state_topic (str):
                servo_state_topic (str):
                speed_to_erpm_offset (str): Offset conversion param from rpm to speed
                speed_to_erpm_gain (float): Gain conversion param from rpm to speed
                steering_to_servo_offset (float): Offset conversion param from servo position to steering angle
                steering_to_servo_gain (float): Gain conversion param from servo position to steering angle
                car_length (float)
        """
        super().__init__("kinematic_car_motion_model")

        self.particles = particles

        required_keyword_args = {
            "motor_state_topic",
            "servo_state_topic",
            "speed_to_erpm_offset",
            "speed_to_erpm_gain",
            "steering_to_servo_offset",
            "steering_to_servo_gain",
            "car_length",
        }
        if not required_keyword_args.issubset(set(kwargs)):
            raise ValueError("Missing required keyword argument")

        # pdb.set_trace()

        self.__dict__.update(kwargs)

        self.state_lock = state_lock or Lock()
        if noise_params is None:
            noise_params = {}

        self.motion_model = KinematicCarMotionModel(self.car_length, **noise_params)

        self.last_servo_cmd = None
        self.last_vesc_stamp = None
        self.initialized = False

        self.servo_subscriber = self.create_subscription(
            Float64,
            self.servo_state_topic,
            self.servo_callback,
            1,
        )

        self.motion_subscriber = self.create_subscription(
            VescStateStamped,
            self.motor_state_topic,
            self.motion_callback,
            1,
        )

    def start(self):
        self.initialized = True

    def servo_callback(self, msg):
        """Caching steering angle data for later use from the servo message."""
        self.last_servo_cmd = msg.data

    def motion_callback(self, msg):
        """Apply the motion model to the particles.

        Converts raw msgs into controls and updates the particles in-place.
        """

        if self.last_servo_cmd is None:
            self.get_logger().warning("No servo command received")
            return

        if self.last_vesc_stamp is None:
            self.get_logger().info("Motion information received for the first time")
            self.last_vesc_stamp = msg.header.stamp
            return

        if not self.initialized:
            return

        curr_speed = (msg.state.speed - self.speed_to_erpm_offset) / self.speed_to_erpm_gain
        curr_steering_angle = (self.last_servo_cmd - self.steering_to_servo_offset) / self.steering_to_servo_gain

        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        last_time = self.last_vesc_stamp.sec + self.last_vesc_stamp.nanosec * 1e-9
        dt = curr_time - last_time

        # self.get_logger().info(f"speed={curr_speed:.4f}, steer={curr_steering_angle:.4f}, dt={dt:.4f}")

        # Reject bad timestamps
        if dt <= 0.0:
            self.last_vesc_stamp = msg.header.stamp
            return

        # Deadband to suppress idle noise
        if abs(curr_speed) < 0.02:
            curr_speed = 0.0

        if abs(curr_steering_angle) < 0.02:
            curr_steering_angle = 0.0

        # If effectively stationary, do not update particles
        if curr_speed == 0.0 and curr_steering_angle == 0.0:
            self.last_vesc_stamp = msg.header.stamp
            return

        with self.state_lock:
            self.motion_model.apply_motion_model(
                self.particles,
                curr_speed,
                curr_steering_angle,
                dt,
            )

        self.last_vesc_stamp = msg.header.stamp

    # def motion_callback(self, msg):
    #     """Apply the motion model to the particles.

    #     Converts raw msgs into controls and updates the particles in-place.
    #     """

    #     # self.get_logger().info(" got got ----------")

    #     if self.last_servo_cmd is None:
    #         self.get_logger().warning("No servo command received")
    #         return

    #     if self.last_vesc_stamp is None:
    #         self.get_logger().info("Motion information received for the first time")
    #         self.last_vesc_stamp = msg.header.stamp
    #         return

    #     if not self.initialized:
    #         return

    #     curr_speed = (msg.state.speed - self.speed_to_erpm_offset) / self.speed_to_erpm_gain

    #     curr_steering_angle = (self.last_servo_cmd - self.steering_to_servo_offset) / self.steering_to_servo_gain

    #     curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    #     last_time = self.last_vesc_stamp.sec + self.last_vesc_stamp.nanosec * 1e-9
    #     dt = curr_time - last_time

    #     print(curr_speed, curr_steering_angle, dt)

    #     with self.state_lock:
    #         self.motion_model.apply_motion_model(
    #             self.particles,
    #             curr_speed,
    #             curr_steering_angle,
    #             dt,
    #         )

    #     self.last_vesc_stamp = msg.header.stamp


def main(args=None):
    rclpy.init(args=args)

    particles = np.zeros((100, 3), dtype=float)

    node = KinematicCarMotionModelROS(
        particles=particles,
        noise_params=None,
        state_lock=None,
        motor_state_topic="vesc/sensors/core",
        servo_state_topic="vesc/sensors/servo_position_comman",
        speed_to_erpm_offset=0.0,
        speed_to_erpm_gain=1.0,
        steering_to_servo_offset=0.0,
        steering_to_servo_gain=1.0,
        car_length=0.33,
    )

    node.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
