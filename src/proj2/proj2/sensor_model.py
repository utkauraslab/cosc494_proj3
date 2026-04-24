#!/usr/bin/env python3
from __future__ import division

from threading import Lock
import numpy as np
import range_libc

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan

OFFSET_FLIP_LIDAR = 0.0


class SingleBeamSensorModel:
    """The single laser beam sensor model."""

    def __init__(self, **kwargs):
        """Initialize the single-beam sensor model.

        Args:
            **kwargs (object): any number of optional keyword arguments:
                hit_std (float): Noise value for hit reading
                z_hit (float): Weight for hit reading
                z_short (float): Weight for short reading
                z_max (float): Weight for max reading
                z_rand (float): Weight for random reading
        """
        defaults = {
            "hit_std": 1.0,
            "z_hit": 0.5,
            "z_short": 0.05,
            "z_max": 0.05,
            "z_rand": 0.5,
        }
        if not set(kwargs).issubset(set(defaults)):
            raise ValueError("Invalid keyword argument provided")
        # These next two lines set the instance attributes from the defaults and
        # kwargs dictionaries. For example, the key "hit_std" becomes the
        # instance attribute self.hit_std.
        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)

        if self.z_short == 0 and self.z_max == 0 and self.z_rand == 0 and self.z_hit == 0:
            raise ValueError("The model is undefined for the given parameters." "You must provide a non-0 value for at least one portion of the model.")

    def precompute_sensor_model(self, max_r):
        """Precompute sensor model probabilities for all pairs of simulated and observed
        distance measurements.

        The probabilities are stored in a 2D array, where the element at index
        (r, d) is the probability of observing measurement r when the simulated
        (expected) measurement is d.

        You will need to normalize the table to ensure probabilities sum to 1, i.e.
        sum P(r | d) over all r should be 1, for all d.

        Args:
            max_r (int): The maximum range (in pixels)

        Returns:
            prob_table: np.array of shape (max_r+1, max_r+1) containing
                the sensor probabilities P(r | d), or P(z_t^k | z_t^k*) from lecture.
        """
        table_width = int(max_r) + 1
        prob_table = np.zeros((table_width, table_width))

        # Get matrices of the same shape as prob_table,
        # where each entry holds the real measurement r (obs_r)
        # or the simulated (expected) measurement d (sim_r).
        obs_r, sim_r = np.mgrid[0:table_width, 0:table_width]

        # Use obs_r and sim_r to vectorize the sensor model precomputation.
        diff = sim_r - obs_r

        # BEGIN QUESTION 2.1
        prob_table = np.zeros_like(prob_table, dtype=np.float64)

        # p_hit(z | z*) = N(z; z*, sigma_hit^2), for 0 <= z <= z_max
        if self.hit_std > 0:
            p_hit = 1.0 / np.sqrt(2.0 * np.pi * self.hit_std**2) * np.exp(-0.5 * (diff / self.hit_std) ** 2)
        else:
            p_hit = np.zeros_like(prob_table, dtype=np.float64)
            p_hit[obs_r == sim_r] = 1.0

        # p_short(z | z*) = 2 * (z* - z) / (z*)^2, for 0 <= z < z*
        p_short = np.zeros_like(prob_table, dtype=np.float64)
        valid_short = obs_r < sim_r
        p_short[valid_short] = 2.0 * (sim_r[valid_short] - obs_r[valid_short]) / (sim_r[valid_short] ** 2)

        # p_max(z | z*) = I(z = z_max)
        p_max = np.zeros_like(prob_table, dtype=np.float64)
        p_max[obs_r == max_r] = 1.0

        # p_rand(z | z*) = 1 / z_max, for 0 <= z < z_max
        p_rand = np.zeros_like(prob_table, dtype=np.float64)
        if max_r > 0:
            p_rand[obs_r < max_r] = 1.0 / max_r

        prob_table = self.z_hit * p_hit + self.z_short * p_short + self.z_max * p_max + self.z_rand * p_rand

        col_sums = prob_table.sum(axis=0)
        valid = col_sums > 0
        prob_table[:, valid] /= col_sums[valid]
        # prob_table[:, ~valid] = np.nan
        prob_table[:, ~valid] = 0.0
        # END QUESTION 2.1

        return prob_table


class LaserScanSensorModelROS(Node):
    """A ROS subscriber that reweights particles according to the sensor model.

    This applies the sensor model to the particles whenever it receives a
    message from the laser scan topic.

    These implementation details can be safely ignored, although you're welcome
    to continue reading to better understand how the entire state estimation
    pipeline is connected.
    """

    def __init__(self, particles, weights, sensor_params=None, state_lock=None, **kwargs):
        """Initialize the laser scan sensor model ROS subscriber.

        Args:
            particles: the particles to update
            weights: the weights to update
            sensor_params: a dictionary of parameters for the sensor model
            state_lock: guarding access to the particles and weights during update,
                since both are shared variables with other processes
            **kwargs: Required unless marked otherwise
                laser_ray_step (int): Step for downsampling laser scans
                exclude_max_range_rays (bool): Whether to exclude rays that are
                    beyond the max range
                map_msg (nav_msgs.msg.MapMetaData): Map metadata to use
                car_length (float): the length of the car
                theta_discretization (int): Discretization of scanning angle. Optional
                inv_squash_factor (float): Make the weight distribution less peaked

        """
        super().__init__("laser_scan_sensor_model")

        if not particles.shape[0] == weights.shape[0]:
            raise ValueError("Must have same number of particles and weights")
        self.particles = particles
        self.weights = weights
        required_keyword_args = {
            "laser_ray_step",
            "exclude_max_range_rays",
            "max_range_meters",
            "map_msg",
            "car_length",
            "scan_topic",
        }
        if not required_keyword_args.issubset(set(kwargs)):
            raise ValueError("Missing required keyword argument")
        defaults = {
            "theta_discretization": 112,
            "inv_squash_factor": 1.0,
        }
        # These next two lines set the instance attributes from the defaults and
        # kwargs dictionaries.
        self.__dict__.update(defaults)
        self.__dict__.update(kwargs)

        self.half_car_length = self.car_length / 2
        self.state_lock = state_lock or Lock()
        sensor_params = {} if sensor_params is None else sensor_params
        sensor_model = SingleBeamSensorModel(**sensor_params)

        # Create map
        o_map = range_libc.PyOMap(self.map_msg)
        # the max range in pixels of the laser
        self.max_range_px = int(self.max_range_meters / self.map_msg.info.resolution)
        self.range_method = range_libc.PyCDDTCast(o_map, self.max_range_px, self.theta_discretization)
        # Alternative range method that can be used for ray casting
        # self.range_method = range_libc.PyRayMarchingGPU(o_map, max_range_px)
        self.range_method.set_sensor_model(sensor_model.precompute_sensor_model(self.max_range_px))

        self.queries = None
        self.ranges = None
        self.laser_angles = None  # The angles of each ray
        self.do_resample = False  # Set for outside code to know when to resample

        self.initialized = False
        self.last_laser = None

        # Subscribe to laser scans
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.lidar_callback,
            1,
        )

    def start(self):
        self.initialized = True

    def lidar_callback(self, msg):
        """Apply the sensor model to downsampled laser measurements.

        Args:
            msg: a sensor_msgs/LaserScan message
        """
        ranges_np = np.asarray(msg.ranges, dtype=np.float32)

        # Initialize angles
        if self.laser_angles is None:
            # self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges_np)) + OFFSET_FLIP_LIDAR
            self.laser_angles = (msg.angle_min + np.arange(len(ranges_np), dtype=np.float32) * msg.angle_increment) + OFFSET_FLIP_LIDAR

        if not self.initialized:
            return

        ranges, angles = self.downsample(ranges_np)

        # Acquire the lock that synchronizes access to the particles. This is
        # necessary because self.particles is shared by the other particle
        # filter classes.
        #
        # The with statement automatically acquires and releases the lock.
        # See the Python documentation for more information:
        # https://docs.python.org/3/library/threading.html#using-locks-conditions-and-semaphores-in-the-with-statement
        with self.state_lock:
            self.apply_sensor_model(ranges, angles)

            total = np.sum(self.weights)

            if not np.isfinite(total) or total <= 1e-12:
                self.weights[:] = 1.0 / self.weights.shape[0]
            else:
                self.weights[:] /= total

            # self.weights /= np.sum(self.weights)

        self.last_laser = msg
        self.do_resample = True

    def apply_sensor_model(self, obs_ranges, obs_angles):
        """Apply the sensor model to self.weights based on the observed laser scan.

        Args:
            obs_ranges: observed distance measurements in meters
            obs_angles: observed laser beam angles in radians
        """
        num_rays = obs_angles.shape[0]
        num_particles = self.particles.shape[0]

        if num_rays == 0 or num_particles == 0:
            return

        # Initialize reusable buffers
        if self.queries is None or self.queries.shape[0] != num_particles:
            self.queries = np.zeros((num_particles, 3), dtype=np.float32)

        if self.ranges is None or self.ranges.shape[0] != num_rays * num_particles:
            self.ranges = np.zeros(num_rays * num_particles, dtype=np.float32)

        # Use particle poses as ray-casting queries.
        # For debugging, do NOT add an extra lidar offset here unless you are sure
        # the particle pose is at the rear axle / car center and the laser is offset.
        self.queries[:, :] = self.particles[:, :].astype(np.float32)
        self.queries[:, 0] += self.half_car_length * np.cos(self.queries[:, 2])
        self.queries[:, 1] += self.half_car_length * np.sin(self.queries[:, 2])

        # Raycast expected measurements.
        # range_libc returns ranges in PIXELS for this map representation.
        self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

        # Convert observed scan ranges from METERS to PIXELS so units match.
        obs_ranges_px = obs_ranges / self.map_msg.info.resolution
        obs_ranges_px = np.clip(obs_ranges_px, 0, self.max_range_px).astype(np.float32)

        # # Optional debug prints:
        # self.get_logger().info(
        #     f"obs px min={np.min(obs_ranges_px):.3f}, max={np.max(obs_ranges_px):.3f}, "
        #     f"expected px min={np.min(self.ranges):.3f}, max={np.max(self.ranges):.3f}"
        # )

        # Reset weights before evaluating sensor model.
        # range_libc writes likelihoods into self.weights.
        self.weights.fill(1.0)

        # Evaluate the sensor model.
        self.range_method.eval_sensor_model(
            obs_ranges_px,
            self.ranges,
            self.weights,
            num_rays,
            num_particles,
        )

        # Optional weight squashing.
        # For debugging / convergence tuning, keep this at 1.0 or disable entirely.
        if self.inv_squash_factor != 1.0:
            np.power(self.weights, self.inv_squash_factor, self.weights)

        self.get_logger().info(
            f"weights raw min={np.min(self.weights):.3e}, " f"max={np.max(self.weights):.3e}, " f"ratio={np.max(self.weights)/(np.min(self.weights)+1e-12):.3e}"
        )

    # def apply_sensor_model(self, obs_ranges, obs_angles):
    #     """Apply the sensor model to self.weights based on the observed laser scan.

    #     Args:
    #         obs_ranges: observed distance measurements
    #         obs_angles: observed laser beam angles
    #     """
    #     num_rays = obs_angles.shape[0]

    #     # Initialize buffer
    #     num_particles = self.particles.shape[0]
    #     if self.queries is None:
    #         self.queries = np.zeros((num_particles, 3), dtype=np.float32)
    #     if self.ranges is None:
    #         self.ranges = np.zeros(num_rays * num_particles, dtype=np.float32)

    #     self.queries[:, :] = self.particles[:, :]
    #     self.queries[:, 0] += self.half_car_length * np.cos(self.queries[:, 2])
    #     self.queries[:, 1] += self.half_car_length * np.sin(self.queries[:, 2])

    #     # Raycasting to get expected measurements
    #     self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

    #     # Evaluate the sensor model
    #     self.range_method.eval_sensor_model(obs_ranges, self.ranges, self.weights, num_rays, self.particles.shape[0])

    #     self.get_logger().info(
    #         f"weights raw min={np.min(self.weights):.3e}, " f"max={np.max(self.weights):.3e}, " f"ratio={np.max(self.weights)/(np.min(self.weights)+1e-12):.3e}"
    #     )

    #     # Squash weights to prevent too much peakiness
    #     np.power(self.weights, self.inv_squash_factor, self.weights)

    def downsample(self, ranges):
        """Downsample the laser rays.

        Args:
            ranges: all observed distance measurements

        Returns:
            ranges: downsampled observed distance measurements
            angles: downsampled observed laser beam angles
        """
        if not self.exclude_max_range_rays:
            angles = np.copy(self.laser_angles[0 :: self.laser_ray_step]).astype(np.float32)
            sampled = ranges[:: self.laser_ray_step].copy()
            sampled[np.isnan(sampled)] = self.max_range_meters
            sampled[np.abs(sampled) < 1e-3] = self.max_range_meters
            return sampled.astype(np.float32), angles

        # We're trying to avoid copying the ranges here, so
        # we silence errors from comparison to NaN instead of overriding these values
        with np.errstate(invalid="ignore"):
            valid_indices = np.logical_and(~np.isnan(ranges), ranges > 0.01, ranges < self.max_range_meters)
        filtered_ranges = ranges[valid_indices]
        filtered_angles = self.laser_angles[valid_indices]

        # Grab expected number of rays
        ray_count = int(self.laser_angles.shape[0] / self.laser_ray_step)
        num_valid = filtered_angles.shape[0]

        if num_valid == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        sample_indices = np.arange(0, num_valid, float(num_valid) / ray_count).astype(int)
        sample_indices = np.clip(sample_indices, 0, num_valid - 1)

        angles = np.copy(filtered_angles[sample_indices]).astype(np.float32)
        ranges = np.copy(filtered_ranges[sample_indices]).astype(np.float32)
        return ranges, angles


def main(args=None):
    rclpy.init(args=args)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
