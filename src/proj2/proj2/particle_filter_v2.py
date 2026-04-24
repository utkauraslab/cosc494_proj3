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


OFFSET_FLIP_LIDAR = np.pi


class ParticleInitializer:
    def __init__(self, x_std=0.1, y_std=0.1, theta_std=0.2):
        self.x_std = x_std
        self.y_std = y_std
        self.theta_std = theta_std

    def reset_click_pose(self, msg, particles, weights):
        """Initialize particles and weights in-place around the clicked pose."""
        n_particles = particles.shape[0]
        theta = utils.quaternion_to_angle(msg.orientation)

        particles[:, 0] = np.random.normal(msg.position.x, self.x_std, n_particles)
        particles[:, 1] = np.random.normal(msg.position.y, self.y_std, n_particles)
        particles[:, 2] = np.random.normal(theta, self.theta_std, n_particles)
        weights[:] = 1.0 / n_particles


class ParticleFilter(Node):
    """The particle filtering state estimation algorithm."""

    def __init__(self):
        super().__init__("particle_filter")

        self._load_parameters()

        self.particle_indices = np.arange(self.n_particles)
        self.particles = np.zeros((self.n_particles, 3), dtype=float)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        self.particle_initializer = ParticleInitializer()

        self.state_lock = Lock()
        self._tf_buffer = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self._tf_buffer, self)

        map_msg = self._get_map()

        self.map_info = map_msg.info
        self.get_logger().info(
            "Received map with resolution "
            f"{self.map_info.resolution} and dimensions "
            f"{self.map_info.width}x{self.map_info.height}"
        )

        map_array = np.array(map_msg.data, dtype=np.int16).reshape(
            (self.map_info.height, self.map_info.width)
        )
        self.permissible_region = np.zeros_like(map_array, dtype=bool)
        self.permissible_region[map_array == 0] = True
        self.permissible_y, self.permissible_x = np.where(self.permissible_region)

        self.pose_pub = self.create_publisher(PoseStamped, "inferred_pose", 1)
        self.particle_pub = self.create_publisher(PoseArray, "particles", 1)
        self.pub_odom = self.create_publisher(Odometry, "odom", 1)

        self.resampler = LowVarianceSampler(self.particles, self.weights, self.state_lock)

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

        sensor_model_worker_params = {
            "laser_ray_step": self.laser_ray_step,
            "exclude_max_range_rays": self.exclude_max_range_rays,
            "max_range_meters": self.max_range_meters,
            "car_length": self.car_length,
            "scan_topic": self.scan_topic,
            "map_msg": map_msg,
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

        self.get_logger().info(f"Waiting for laser scan on {self.scan_topic}")
        scan_msg = self._wait_for_message(
            self.scan_topic,
            LaserScan,
            timeout_sec=10.0,
        )
        if scan_msg is None:
            raise RuntimeError(f"Timed out waiting for {self.scan_topic}")

        self._viz_thread = Thread(target=self.visualize, daemon=True)
        self._viz_thread.start()

        if self.publish_tf:
            self.get_logger().info("Starting TF broadcaster")
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            self._tf_pub_thread = Thread(target=self._publish_tf, daemon=True)
            self._tf_pub_thread.start()

        initialization = [self.initial_x, self.initial_y, self.initial_theta]
        if not any(map(np.isnan, initialization)):
            self.set_pose(*initialization)
        elif not all(map(np.isnan, initialization)):
            self.get_logger().error(
                "Initial pose estimate "
                f"({self.initial_x}, {self.initial_y}, {self.initial_theta}) "
                "is partially specified. You must provide x, y and theta components."
            )

        self.get_logger().info("Startup complete. Waiting for initial pose estimate")

    def _declare_and_get(self, name, default_value):
        self.declare_parameter(name, default_value)
        return self.get_parameter(name).value

    def _load_parameters(self):
        self.publish_tf = bool(self._declare_and_get("publish_tf", True))
        self.tf_prefix = str(self._declare_and_get("tf_prefix", ""))
        self.n_particles = int(self._declare_and_get("n_particles", 500))
        self.n_viz_particles = int(self._declare_and_get("n_viz_particles", 100))
        self.car_length = float(self._declare_and_get("kinematics.car_length", 0.33))

        self.laser_ray_step = int(self._declare_and_get("laser_ray_step", 1))
        self.exclude_max_range_rays = bool(
            self._declare_and_get("exclude_max_range_rays", False)
        )
        self.max_range_meters = float(
            self._declare_and_get("max_range_meters", 10.0)
        )

        self.speed_to_erpm_offset = float(
            self._declare_and_get("vesc.speed_to_erpm_offset", 0.0)
        )
        self.speed_to_erpm_gain = float(
            self._declare_and_get("vesc.speed_to_erpm_gain", 4350.0)
        )
        self.steering_to_servo_offset = float(
            self._declare_and_get("vesc.steering_angle_to_servo_offset", 0.5)
        )
        self.steering_to_servo_gain = float(
            self._declare_and_get("vesc.steering_angle_to_servo_gain", -1.2135)
        )

        self.use_map_topic = bool(self._declare_and_get("use_map_topic", False))
        self.motor_state_topic = str(
            self._declare_and_get("motor_state_topic", "vesc/sensors/core")
        )
        self.servo_state_topic = str(
            self._declare_and_get(
                "servo_state_topic",
                "vesc/sensors/servo_position_command",
            )
        )
        self.scan_topic = str(self._declare_and_get("scan_topic", "scan"))

        self.motion_params = {
            "vel_std": float(self._declare_and_get("motion_params.vel_std", 0.1)),
            "delta_std": float(
                self._declare_and_get("motion_params.delta_std", 0.5)
            ),
            "x_std": float(self._declare_and_get("motion_params.x_std", 0.05)),
            "y_std": float(self._declare_and_get("motion_params.y_std", 0.05)),
            "theta_std": float(
                self._declare_and_get("motion_params.theta_std", 0.05)
            ),
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
            self.get_logger().info("Waiting for map topic /map")
            map_qos = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            )
            map_msg = self._wait_for_message(
                "/map",
                OccupancyGrid,
                timeout_sec=10.0,
                qos_profile=map_qos,
            )
            if map_msg is None:
                raise RuntimeError("Timed out waiting for /map")
            return map_msg

        self.get_logger().info("Waiting for /map_server/map service")
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
            self.get_logger().info(f"Setting pose to {pose_msg}")
            self.particle_initializer.reset_click_pose(
                pose_msg,
                self.particles,
                self.weights,
            )
            self.sensor_model.start()
            self.motion_model.start()

    def spin(self):
        while rclpy.ok():
            time.sleep(1.0 / 50.0)
            if self.sensor_model.do_resample:
                self.sensor_model.do_resample = False
                self.resampler.resample()

    def expected_pose(self):
        """Compute the expected state from the current particles and weights."""
        cosines = np.cos(self.particles[:, 2])
        sines = np.sin(self.particles[:, 2])
        theta = np.arctan2(np.dot(sines, self.weights), np.dot(cosines, self.weights))
        position = np.dot(self.particles[:, 0:2].transpose(), self.weights)

        position[0] += (self.car_length / 2.0) * np.cos(theta)
        position[1] += (self.car_length / 2.0) * np.sin(theta)
        return np.array((position[0], position[1], theta), dtype=float)

    def clicked_pose_cb(self, msg):
        self.set_pose_msg(msg.pose.pose)

    def visualize(self):
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
        pa = PoseArray()
        pa.header = utils.make_header("map")
        pa.poses = utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def _infer_pose(self):
        with self.state_lock:
            inferred_pose = self.expected_pose()

        if not isinstance(inferred_pose, np.ndarray):
            return None

        ps = PoseStamped()
        ps.header = utils.make_header("map")
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = inferred_pose[0]
        ps.pose.position.y = inferred_pose[1]
        ps.pose.orientation = utils.angle_to_quaternion(inferred_pose[2])
        return ps

    def _publish_tf(self):
        while rclpy.ok():
            time.sleep(1.0 / 40.0)

            if not self.sensor_model.initialized or not self.motion_model.initialized:
                continue

            pose = self._infer_pose()
            if pose is None:
                continue

            pa = utils.pose_to_particle(pose.pose)
            future_time = self.get_clock().now() + Duration(seconds=0.1)
            pose.header.stamp = future_time.to_msg()

            try:
                odom_to_laser = self._tf_buffer.lookup_transform(
                    self.tf_prefix + "laser_link",
                    self.tf_prefix + "odom",
                    Time(),
                    timeout=Duration(seconds=0.1),
                )
            except Exception as e:
                self.get_logger().error(str(e))
                self.get_logger().error("failed to find odom")
                continue

            delta_off, delta_rot = utils.transform_stamped_to_pq(odom_to_laser)
            pa[2] += OFFSET_FLIP_LIDAR

            off_x = delta_off[0] * np.cos(pa[2]) - delta_off[1] * np.sin(pa[2])
            off_y = delta_off[0] * np.sin(pa[2]) + delta_off[1] * np.cos(pa[2])

            transform = TransformStamped()
            transform.header.stamp = pose.header.stamp
            transform.header.frame_id = "map"
            transform.child_frame_id = self.tf_prefix + "odom"
            transform.transform.translation.x = pa[0] + off_x
            transform.transform.translation.y = pa[1] + off_y
            transform.transform.translation.z = 0.0

            yaw = pa[2] + euler_from_quaternion(delta_rot)[2]
            nq = quaternion_from_euler(0.0, 0.0, yaw)
            transform.transform.rotation.x = nq[0]
            transform.transform.rotation.y = nq[1]
            transform.transform.rotation.z = nq[2]
            transform.transform.rotation.w = nq[3]

            self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)

    node = ParticleFilter()
    pf_thread = Thread(target=node.spin, daemon=True)
    pf_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
