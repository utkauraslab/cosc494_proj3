#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

import tf_transformations  # provides quaternion math


def _q_to_list(q):
    return [q.x, q.y, q.z, q.w]


def _list_to_q(lst, qmsg):
    qmsg.x, qmsg.y, qmsg.z, qmsg.w = map(float, lst)
    return qmsg


class FakeLocalization(Node):
    """
    ROS2 replacement for ROS1 fake_localization.

    Subscribes to a "ground-truth" Odometry message that represents base_link pose in the map frame
    and publishes TF map->odom by combining with the current odom->base_link TF.

    This connects TF trees: map -> odom -> base_link.
    """

    def __init__(self):
        super().__init__("fake_localization")

        # -------- Parameters --------
        self.declare_parameter("ground_truth_odom_topic", "/car/odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tf_timeout_sec", 0.2)

        self.gt_topic = self.get_parameter("ground_truth_odom_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        # -------- TF --------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # -------- Sub --------
        self.sub = self.create_subscription(Odometry, self.gt_topic, self._gt_cb, 10)

        self.get_logger().info(
            f"FakeLocalization started.\n"
            f"  ground_truth_odom_topic: {self.gt_topic}\n"
            f"  publishing TF: {self.map_frame} -> {self.odom_frame}\n"
            f"  using base_frame: {self.base_frame}\n"
            f"  expects TF available: {self.odom_frame} -> {self.base_frame}"
        )

        self._warned_missing_tf = False

    def _gt_cb(self, msg: Odometry):
        """
        msg.pose.pose is interpreted as T_map_base (ground truth base pose in map frame).
        We also need T_odom_base from TF to compute T_map_odom = T_map_base * inv(T_odom_base).
        """
        # --- Build T_map_base from msg ---
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        T_map_base = tf_transformations.quaternion_matrix(_q_to_list(q))
        T_map_base[0, 3] = float(p.x)
        T_map_base[1, 3] = float(p.y)
        T_map_base[2, 3] = float(p.z)

        # --- Lookup T_odom_base from TF ---
        try:
            # Use latest available TF. If you want time-synced, use msg.header.stamp -> rclpy Time.
            tf_odom_base = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.base_frame,
                rclpy.time.Time(),  # latest
            )
        except Exception as e:
            if not self._warned_missing_tf:
                self.get_logger().warn(
                    f"Waiting for TF {self.odom_frame} -> {self.base_frame}. " f"Cannot publish {self.map_frame}->{self.odom_frame} yet. Error: {e}"
                )
                self._warned_missing_tf = True
            return

        self._warned_missing_tf = False

        t = tf_odom_base.transform.translation
        r = tf_odom_base.transform.rotation

        T_odom_base = tf_transformations.quaternion_matrix(_q_to_list(r))
        T_odom_base[0, 3] = float(t.x)
        T_odom_base[1, 3] = float(t.y)
        T_odom_base[2, 3] = float(t.z)

        # --- Compute T_map_odom ---
        T_base_odom = tf_transformations.inverse_matrix(T_odom_base)
        T_map_odom = T_map_base @ T_base_odom

        # --- Publish TF map -> odom ---
        out = TransformStamped()
        out.header.stamp = msg.header.stamp  # keep consistent with GT msg time
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.odom_frame

        out.transform.translation.x = float(T_map_odom[0, 3])
        out.transform.translation.y = float(T_map_odom[1, 3])
        out.transform.translation.z = float(T_map_odom[2, 3])

        q_out = tf_transformations.quaternion_from_matrix(T_map_odom)
        _list_to_q(q_out, out.transform.rotation)

        self.tf_broadcaster.sendTransform(out)


def main():
    rclpy.init()
    node = FakeLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
