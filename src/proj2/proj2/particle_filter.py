#!/usr/bin/env python3
from __future__ import division
from threading import Lock, Thread
import time
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
import tf2_ros

from geometry_msgs.msg import (
    Pose,
    PoseArray,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap
from tf_transformations import quaternion_from_euler, euler_from_quaternion

from proj2 import utils
from proj2.motion_model import KinematicCarMotionModelROS
from proj2.sensor_model import LaserScanSensorModelROS
from proj2.resampler import LowVarianceSampler
import pdb

OFFSET_FLIP_LIDAR = 0.0


class ParticleInitializer:
    def __init__(self, x_std=0.1, y_std=0.1, theta_std=0.2):
        self.x_std = x_std
        self.y_std = y_std
        self.theta_std = theta_std

    def reset_click_pose(self, msg, particles, weights):
        """Initialize the particles and weights in-place.

        The particles should be sampled from a Gaussian distribution around the
        initial pose. Remember to write vectorized code.

        Args:
            msg: a geometry_msgs/Pose message with the initial pose
            particles: the particles to initialize
            weights: the weights associated with particles
        """
        n_particles = particles.shape[0]
        # Hint: use utils.quaternion_to_angle to compute the orientation theta.
        # BEGIN QUESTION 3.1

        theta = utils.quaternion_to_angle(msg.orientation)

        particles[:, 0] = np.random.normal(msg.position.x, self.x_std, n_particles)
        particles[:, 1] = np.random.normal(msg.position.y, self.y_std, n_particles)
        particles[:, 2] = np.random.normal(theta, self.theta_std, n_particles)

        weights[:] = 1.0 / n_particles

        # END QUESTION 3.1


class ParticleFilter(Node):
    """The particle filtering state estimation algorithm.

    These implementation details can be safely ignored, although you're welcome
    to continue reading to better understand how the entire state estimation
    pipeline is connected.
    """

    def __init__(self):
        """Initialize the particle filter.

        ROS2 version:
            The node declares and loads its own parameters directly.
        """
        super().__init__("particle_filter")

        self._load_parameters()

        # Cached list of particle indices
        self.particle_indices = np.arange(self.n_particles)
        # Numpy matrix of dimension N_PARTICLES x 3
        self.particles = np.zeros((self.n_particles, 3), dtype=float)
        # Numpy matrix containing weight for each particle
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        self.particle_initializer = ParticleInitializer()

        self.state_lock = Lock()
        self._tf_buffer = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self._tf_buffer, self)

        map_msg = self._get_map()

        self.map_info = map_msg.info
        self.get_logger().info(
            "Received map with resolution {} and dimensions {}x{}".format(
                self.map_info.resolution,
                self.map_info.width,
                self.map_info.height,
            )
        )

        # Permissible region
        array_255 = np.array(map_msg.data, dtype=np.int16).reshape((self.map_info.height, self.map_info.width))
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255 == 0] = True
        self.permissible_y, self.permissible_x = np.where(self.permissible_region)

        # Publishes the expected pose
        self.pose_pub = self.create_publisher(PoseStamped, "inferred_pose", 1)
        # Publishes a subsample of the particles
        self.particle_pub = self.create_publisher(PoseArray, "particles", 1)
        # Publishes the path of the car
        self.pub_odom = self.create_publisher(Odometry, "odom", 1)

        # Outside caller can use this resampler to resample particles
        self.resampler = LowVarianceSampler(self.particles, self.weights, self.state_lock)

        # Initialize the motion model subscriber
        motion_model_worker_params = {
            "car_length": self.car_length,
            "speed_to_erpm_offset": self.speed_to_erpm_offset,
            "speed_to_erpm_gain": self.speed_to_erpm_gain,
            "steering_to_servo_offset": self.steering_to_servo_offset,
            "steering_to_servo_gain": self.steering_to_servo_gain,
            "motor_state_topic": self.motor_state_topic,
            "servo_state_topic": self.servo_state_topic,
        }

        self.motion_model = KinematicCarMotionModelROS(
            self.particles,
            noise_params=self.motion_params,
            state_lock=self.state_lock,
            **motion_model_worker_params,
        )

        # Initialize the sensor model subscriber
        sensor_model_worker_params = {
            "laser_ray_step": self.laser_ray_step,
            "exclude_max_range_rays": self.exclude_max_range_rays,
            "max_range_meters": self.max_range_meters,
            "car_length": self.car_length,
            "map_msg": map_msg,
            "scan_topic": self.scan_topic,
        }
        self.sensor_model = LaserScanSensorModelROS(
            self.particles,
            self.weights,
            sensor_params=self.sensor_params,
            state_lock=self.state_lock,
            **sensor_model_worker_params,
        )

        self.click_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.clicked_pose_cb,
            1,
        )

        self._viz_timer = Thread(target=self.visualize, daemon=True)
        self._viz_timer.start()

        if self.publish_tf:
            self.get_logger().info("Starting TF broadcaster")
            self._tf_pub_timer = Thread(target=self._publish_tf, daemon=True)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self._tf_pub_timer.start()

        initialization = [self.initial_x, self.initial_y, self.initial_theta]
        if not any(map(np.isnan, initialization)):
            self.set_pose(*initialization)
            self.get_logger().info("initial pose estimate provided: ({}, {}, {})".format(*initialization))
        elif not all(map(np.isnan, initialization)):
            self.get_logger().error(
                "Initial pose estimate ({}, {}, {}) is partially specified. " "You must provide x, y and theta components.".format(*initialization)
            )

        self.get_logger().info(f"Waiting for laser scan on {self.scan_topic}")
        scan_msg = self._wait_for_message(self.scan_topic, LaserScan, timeout_sec=10.0)
        if scan_msg is None:
            raise RuntimeError(f"Timed out waiting for {self.scan_topic}")

        self.get_logger().info("Startup complete. Waiting for initial pose estimate")

    def _declare_and_get(self, name, default_value):
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).value

    def _load_parameters(self):
        """Declare and load ROS2 parameters directly inside the node."""
        self.publish_tf = bool(self._declare_and_get("publish_tf", True))
        self.tf_prefix = str(self._declare_and_get("tf_prefix", ""))
        self.n_particles = int(self._declare_and_get("n_particles", 500))
        self.n_viz_particles = int(self._declare_and_get("n_viz_particles", 100))
        self.car_length = float(self._declare_and_get("kinematics.car_length", 0.33))
        self.laser_ray_step = int(self._declare_and_get("laser_ray_step", 1))
        self.exclude_max_range_rays = bool(self._declare_and_get("exclude_max_range_rays", False))
        self.max_range_meters = float(self._declare_and_get("max_range_meters", 10.0))
        self.speed_to_erpm_offset = float(self._declare_and_get("vesc.speed_to_erpm_offset", 0.0))
        self.speed_to_erpm_gain = float(self._declare_and_get("vesc.speed_to_erpm_gain", 4350.0))
        self.steering_to_servo_offset = float(self._declare_and_get("vesc.steering_angle_to_servo_offset", 0.5))
        self.steering_to_servo_gain = float(self._declare_and_get("vesc.steering_angle_to_servo_gain", -1.2135))
        self.use_map_topic = bool(self._declare_and_get("use_map_topic", False))
        self.motor_state_topic = str(self._declare_and_get("motor_state_topic", "vesc/sensors/core"))
        self.servo_state_topic = str(
            self._declare_and_get(
                "servo_state_topic",
                "vesc/sensors/servo_position_command",
            )
        )
        self.scan_topic = str(self._declare_and_get("scan_topic", "scan"))

        self.motion_params = {
            "vel_std": float(self._declare_and_get("motion_params.vel_std", 0.1)),
            "delta_std": float(self._declare_and_get("motion_params.delta_std", 0.5)),
            "x_std": float(self._declare_and_get("motion_params.x_std", 0.05)),
            "y_std": float(self._declare_and_get("motion_params.y_std", 0.05)),
            "theta_std": float(self._declare_and_get("motion_params.theta_std", 0.05)),
        }

        self.sensor_params = {
            "hit_std": float(self._declare_and_get("sensor_params.hit_std", 1.0)),
            "z_hit": float(self._declare_and_get("sensor_params.z_hit", 0.5)),
            "z_short": float(self._declare_and_get("sensor_params.z_short", 0.1)),
            "z_max": float(self._declare_and_get("sensor_params.z_max", 0.05)),
            "z_rand": float(self._declare_and_get("sensor_params.z_rand", 0.5)),
        }

        self.initial_x = float(self._declare_and_get("initial_x", math.nan))
        self.initial_y = float(self._declare_and_get("initial_y", math.nan))
        self.initial_theta = float(self._declare_and_get("initial_theta", math.nan))

    def _get_map(self):
        if self.use_map_topic:
            # Save info about map
            self.get_logger().info("Waiting for map")

            map_msg = self._wait_for_message(
                "/map",
                OccupancyGrid,
                timeout_sec=10.0,
                qos_profile=QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
            if map_msg is None:
                raise RuntimeError("Timed out waiting for /map")
            return map_msg

        mapsrv = self.create_client(GetMap, "/map_server/map")
        if not mapsrv.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("Timed out waiting for /map_server/map service")
        req = GetMap.Request()
        future = mapsrv.call_async(req)
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
        result = future.result()
        if result is None:
            raise RuntimeError("Failed to get map from /map_server/map")
        return result.map

    def _wait_for_message(self, topic_name, msg_type, timeout_sec=10.0, qos_profile=1):
        msg_box = {"msg": None}

        def cb(msg):
            msg_box["msg"] = msg

        sub = self.create_subscription(msg_type, topic_name, cb, qos_profile)
        start = time.time()
        try:
            while rclpy.ok() and msg_box["msg"] is None:
                rclpy.spin_once(self, timeout_sec=0.1)
                if timeout_sec is not None and (time.time() - start) > timeout_sec:
                    break
        finally:
            self.destroy_subscription(sub)
        return msg_box["msg"]

    def set_pose(self, x, y, theta):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.orientation = utils.angle_to_quaternion(theta)
        self.set_pose_msg(pose)

    def set_pose_msg(self, pose_msg):
        with self.state_lock:
            self.get_logger().info("Setting pose to {}".format(pose_msg))
            self.particle_initializer.reset_click_pose(
                pose_msg,
                self.particles,
                self.weights,
            )
            # They may be waiting on an initial estimate
            self.sensor_model.start()
            self.motion_model.start()

            self.get_logger().info("--------------------------------------------")
            self.get_logger().info("Starting motion and sensor models!")
            self.get_logger().info("--------------------------------------------")

    def spin(self):
        # We're limited by rate of the sensor and the speed data.
        # Sensor is usually slowest at around 10Hz
        while rclpy.ok():  # Keep going until we kill it
            time.sleep(1.0 / 50.0)
            # Check if the sensor model says it's time to resample
            if self.sensor_model.do_resample:
                # Reset so that we don't keep resampling
                self.sensor_model.do_resample = False
                self.resampler.resample()

    def expected_pose(self):
        """Compute the expected state, given current particles and weights.

        Use cosine and sine averaging to more accurately compute average theta.

        To get one combined value, use the dot product of position and weight vectors
        https://en.wikipedia.org/wiki/Mean_of_circular_quantities

        Returns:
            np.array of the expected state
        """
        cosines = np.cos(self.particles[:, 2])
        sines = np.sin(self.particles[:, 2])
        theta = np.arctan2(np.dot(sines, self.weights), np.dot(cosines, self.weights))
        position = np.dot(self.particles[:, 0:2].transpose(), self.weights)

        # Offset to car's center of mass
        position[0] += (self.car_length / 2) * np.cos(theta)
        position[1] += (self.car_length / 2) * np.sin(theta)
        return np.array((position[0], position[1], theta), dtype=float)

    def clicked_pose_cb(self, msg):
        """Reinitialize particles and weights according to the received initial pose.

        Args:
            msg: a geometry_msgs/Pose message with the initial pose
        """
        self.set_pose_msg(msg.pose.pose)

    def visualize(self):
        """Visualize the current state of the particle filter.

        1. Publishes a tf between the map and the laser. Necessary for
           visualizing the laser scan in the map.
        2. Publishes a PoseStamped message with the expected pose of the car.
        3. Publishes a subsample of the particles, where particles with higher
           weights are more likely to be sampled.
        """
        while rclpy.ok():
            inferred_pose = self._infer_pose()
            if inferred_pose is None:
                time.sleep(1.0 / 4.0)
                continue

            self.pose_pub.publish(inferred_pose)

            odom = Odometry()
            odom.header = inferred_pose.header
            odom.pose.pose = inferred_pose.pose
            self.pub_odom.publish(odom)

            with self.state_lock:
                if self.particles.shape[0] > self.n_viz_particles:
                    # randomly downsample particles
                    proposal_indices = np.random.choice(
                        self.particle_indices,
                        self.n_viz_particles,
                        p=self.weights,
                    )
                    self.publish_particles(self.particles[proposal_indices, :])
                else:
                    self.publish_particles(self.particles)
            time.sleep(1.0 / 4.0)

    def publish_particles(self, particles):
        """Publishes a pose array of particles."""
        pa = PoseArray()
        pa.header = utils.make_header("map", node=self)
        pa.poses = utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def _infer_pose(self):
        """Return a geometry_msgs/PoseStamped message with the current expected pose."""
        with self.state_lock:
            inferred_pose = self.expected_pose()
        ps = None
        if isinstance(inferred_pose, np.ndarray):
            ps = PoseStamped()
            ps.header = utils.make_header("map", node=self)
            # This estimate is as current as the particles
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position.x = inferred_pose[0]
            ps.pose.position.y = inferred_pose[1]
            ps.pose.orientation = utils.angle_to_quaternion(inferred_pose[2])
        return ps

    def _publish_tf(self):
        """Publish a transform between map and odom frames."""
        while rclpy.ok():
            time.sleep(1.0 / 40.0)
            if not self.sensor_model.initialized or not self.motion_model.initialized:
                continue
            pose = self._infer_pose()
            if pose is None:
                continue
            pa = utils.pose_to_particle(pose.pose)

            # Future date transform so consumers know that the prediction is
            # good for a little while
            future_time = self.get_clock().now() + Duration(seconds=0.1)
            pose.header.stamp = future_time.to_msg()

            try:
                # Look up the transform between laser and odom
                odom_to_laser = self._tf_buffer.lookup_transform(
                    self.tf_prefix + "laser_link",
                    self.tf_prefix + "odom",
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
            except Exception as e:  # Will occur if odom frame does not exist
                self.get_logger().error(str(e))
                self.get_logger().error("failed to find odom")
                continue

            delta_off, delta_rot = utils.transform_stamped_to_pq(odom_to_laser)
            pa[2] += OFFSET_FLIP_LIDAR

            # Transform offset to be w.r.t the map
            off_x = delta_off[0] * np.cos(pa[2]) - delta_off[1] * np.sin(pa[2])
            off_y = delta_off[0] * np.sin(pa[2]) + delta_off[1] * np.cos(pa[2])

            # Create the transform message
            transform = TransformStamped()
            transform.header.stamp = pose.header.stamp
            transform.header.frame_id = "map"
            transform.child_frame_id = self.tf_prefix + "odom"
            transform.transform.translation.x = pa[0] + off_x
            transform.transform.translation.y = pa[1] + off_y
            transform.transform.translation.z = 0.0
            nq = quaternion_from_euler(
                0.0,
                0.0,
                pa[2] + euler_from_quaternion(delta_rot)[2],
            )
            transform.transform.rotation.x = nq[0]
            transform.transform.rotation.y = nq[1]
            transform.transform.rotation.z = nq[2]
            transform.transform.rotation.w = nq[3]

            # Broadcast the transform
            self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    # rclpy.init(args=args)

    # node = ParticleFilter()
    # pf_thread = Thread(target=node.spin, daemon=True)
    # pf_thread.start()

    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()

    rclpy.init(args=args)

    pf_node = ParticleFilter()

    executor = MultiThreadedExecutor()
    executor.add_node(pf_node)
    executor.add_node(pf_node.motion_model)
    executor.add_node(pf_node.sensor_model)

    pf_thread = Thread(target=pf_node.spin, daemon=True)
    pf_thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        pf_node.motion_model.destroy_node()
        pf_node.sensor_model.destroy_node()
        pf_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
