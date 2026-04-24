#!/usr/bin/env python3
"""
ROS2 geometry and map utility functions.
"""

from __future__ import division

import heapq
import numpy as np

import rclpy
from rclpy.duration import Duration

from geometry_msgs.msg import (
    Pose,
    PoseStamped,
    Transform,
    TransformStamped,
    Quaternion,
    Point32,
)
from nav_msgs.srv import GetMap
from std_msgs.msg import Header


# Prefer ROS2-friendly tf_transformations (pip package), fall back to tf_conversions if present.
try:
    import tf_transformations as transformations  # ROS2-friendly
except Exception:  # pragma: no cover
    from tf_conversions import transformations  # type: ignore


def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np.arrays.

    Args:
      msg: ROS message to be converted

    Returns:
      p: position as a np.array
      q: quaternion as a np.array (order = [x, y, z, w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np.arrays.

    Args:
      msg: ROS message to be converted

    Returns:
      p: position as a np.array
      q: quaternion as a np.array (order = [x, y, z, w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np.arrays.

    Args:
      msg: ROS message to be converted

    Returns:
      p: position as a np.array
      q: quaternion as a np.array (order = [x, y, z, w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np.arrays.

    Args:
      msg: ROS message to be converted

    Returns:
      p: position as a np.array
      q: quaternion as a np.array (order = [x, y, z, w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Convert geometric ROS messages into SE(3).

    Args:
      msg: ROS message to be converted. Acceptable types include
        C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
        C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}

    Returns:
      A 4x4 SE(3) matrix as a np.array

    Raises:
      TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")

    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError("Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm

    g = transformations.quaternion_matrix(q)
    g[0:3, -1] = p
    return g


def angle_to_quaternion(angle):
    """Convert yaw angle into a Quaternion ROS message.

    Args:
      angle: Yaw angle (radians)

    Returns:
      An equivalent C{geometry_msgs/Quaternion} message
    """

    qx, qy, qz, qw = transformations.quaternion_from_euler(0, 0, angle)
    quat = Quaternion()
    quat.x = float(qx)
    quat.y = float(qy)
    quat.z = float(qz)
    quat.w = float(qw)

    return quat


def quaternion_to_angle(q):
    """Convert a C{geometry_msgs/Quaternion} into a yaw angle.

    Args:
      q: ROS message to be converted

    Returns:
      The equivalent yaw angle (radians)
    """
    _, _, yaw = transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))
    return yaw


def rotation_matrix(theta):
    """Construct a rotation matrix from a given angle.

    Args:
      theta: Angle (radians)

    Returns:
      The equivalent 2x2 rotation matrix as a np.array
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def particle_to_pose(particle):
    """Convert a particle to a C{geometry_msgs/Pose} message.

    Args:
      particle: The particle to convert [x, y, theta]

    Returns:
      An equivalent C{geometry_msgs/Pose} message
    """
    pose = Pose()
    pose.position.x = float(particle[0])
    pose.position.y = float(particle[1])
    pose.orientation = angle_to_quaternion(float(particle[2]))
    return pose


def pose_to_particle(msg):
    """Convert a C{geometry_msgs/Pose} into a particle.

    Args:
      msg: ROS message to be converted.

    Returns:
      A particle [x, y, theta]
    """
    x = msg.position.x
    y = msg.position.y
    theta = quaternion_to_angle(msg.orientation)
    return [x, y, theta]


def particles_to_poses(particles):
    """Convert a list of particles to a list of C{geometry_msgs/Pose} messages.

    Args:
      particles: A list of particles, where each element is itself a list of the form [x, y, theta]

    Returns:
      A list of equivalent C{geometry_msgs/Pose} messages
    """
    return list(map(particle_to_pose, particles))


def make_header(frame_id, stamp=None, *, node=None, clock=None):
    """Create a header with the given frame_id and stamp.

    The default value of stamp is None, which results in a stamp denoting the
    time at which this function was called.

    ROS2 note:
      - If stamp is None, you must provide either:
          * node: a rclpy.node.Node, or
          * clock: a rclpy.clock.Clock
        so we can set header.stamp using ROS2 time.

    Args:
      frame_id: The desired coordinate frame
      stamp: The desired stamp (builtin_interfaces/msg/Time)
      node: Optional ROS2 Node (used to get time if stamp is None)
      clock: Optional ROS2 Clock (used to get time if stamp is None)

    Returns:
      The resulting header
    """
    if stamp is None:
        if clock is None and node is not None:
            clock = node.get_clock()
        if clock is None:
            raise ValueError("make_header: provide stamp, or node/clock to generate one.")
        stamp = clock.now().to_msg()

    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def point(npt):
    """Convert x/y coordinates into a C{geometry_msgs/Point32} message.

    Args:
      npt: A list of length two containing x and y coordinates

    Returns:
      A C{geometry_msgs/Point32} message
    """
    pt = Point32()
    pt.x, pt.y = float(npt[0]), float(npt[1])
    return pt


def points(arr):
    """Convert a list of coordinates into a list of equivalent point messages.

    Args:
      arr: A list of coordinates, where each element is itself a list of the form [x, y]

    Returns:
      A list of C{geometry_msgs/Point32} messages
    """
    return list(map(point, arr))


def get_map(node, map_topic, timeout_sec=5.0):
    """Get the map from the map server.

    ROS1 used rospy.wait_for_service + ServiceProxy.
    ROS2 requires a Node to create a client and call asynchronously.

    Args:
      node: rclpy Node (required)
      map_topic: The service topic that will provide the map (e.g., '/map' or '/static_map')
      timeout_sec: How long to wait for the service

    Returns:
      map_img: A np.array with dimensions (map_info.height, map_info.width).
        A zero at a particular location indicates that it is impermissible.
        A one indicates that the location is permissible.
      map_info: Info about the map, see official docs for more info.
    """
    client = node.create_client(GetMap, map_topic)

    # Wait for service
    if not client.wait_for_service(timeout_sec=timeout_sec):
        raise RuntimeError(f"Map service '{map_topic}' not available after {timeout_sec}s")

    req = GetMap.Request()
    future = client.call_async(req)

    # Block until response (similar to ROS1 wait_for_service + immediate call)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done() or future.result() is None:
        raise RuntimeError(f"Failed to call map service '{map_topic}'")

    map_msg = future.result().map
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))

    map_img = np.zeros_like(array_255, dtype=bool)
    map_img[array_255 == 0] = 1
    return map_img, map_msg.info


def map_to_world(poses, map_info, out=None):
    """Convert an array of pixel locations in the map to poses in the world.

    Args:
      poses: Pixel poses in the map, converted in place. Should be nx3 np.array.
      map_info: Info about the map (returned by get_map)
      out: Optional output buffer to store results in. Defaults to the poses array
    """
    poses = np.atleast_2d(poses)
    if out is None:
        out = poses
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # Rotation
    out[:, [0, 1]] = np.matmul(poses[:, [0, 1]], rotation_matrix(angle))
    if poses.shape[1] == 3:
        out[:, 2] = poses[:, 2] + angle

    # Scale
    out[:, :2] *= float(scale)

    # Translate
    out[:, 0] += map_info.origin.position.x
    out[:, 1] += map_info.origin.position.y

    return out


def world_to_map(poses, map_info, out=None):
    """Convert an array of poses in the world to pixel locations in the map image.

    Args:
      poses: Poses in the world, converted in place. Should be nx3 np.array.
      map_info: Info about the map (returned by get_map)
      out: Optional output buffer to store results in. Defaults to the poses array
    """
    poses = np.atleast_2d(poses)
    if out is None:
        out = poses
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # Translation
    out[:, 0] = poses[:, 0] - map_info.origin.position.x
    out[:, 1] = poses[:, 1] - map_info.origin.position.y

    # Scale
    out[:, :2] = out[:, :2] * (1.0 / float(scale))

    # Rotate poses
    out[:, [0, 1]] = np.matmul(out[:, [0, 1]], rotation_matrix(angle))
    if poses.shape[1] == 3:
        out[:, 2] = poses[:, 2] + angle

    return out


def estimation_error(estimates, references):
    """Compute the error between the estimated and ground truth states.

    Args:
        estimates: Estimated states (from particle filter)
        references: Ground truth states

    Returns:
        position_errors: Positional errors at each timestep
        abs_angular_error: Absolute angular errors at each timestep
    """
    position_errors = np.linalg.norm(estimates[:, :2] - references[:, :2], axis=1)
    estimate_angles = estimates[:, 2]
    reference_angles = references[:, 2]
    angular_errors = np.arctan2(
        np.sin(reference_angles - estimate_angles),
        np.cos(reference_angles - estimate_angles),
    )
    return position_errors, np.abs(angular_errors)


class PriorityQueue(object):
    def __init__(self):
        self.min_heap = []
        self.pop_counter = 0

    def __len__(self):
        return len(self.min_heap)

    def push(self, elem):
        heapq.heappush(self.min_heap, elem)

    def peek(self):
        if not self.min_heap:
            raise IndexError("no elements to peek")
        return self.min_heap[0][1]

    def pop(self):
        if not self.min_heap:
            raise IndexError("no elements to pop")
        elem = heapq.heappop(self.min_heap)
        self.pop_counter += 1
        return elem
