#!/usr/bin/env python3
# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team
# License: BSD 3-Clause. See LICENSE.md file in root directory.
#
# Modified to ROS2 by Fei Liu, University of Tennessee, 2026
#
# Further modified: keyboard-based mux switching between TELEOP and AUTO
# with STOP-on-switch behavior.

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from ackermann_msgs.msg import AckermannDriveStamped


HELP = """
Terminal Keyboard Teleop + Tele/Auto Switch (AckermannDriveStamped)

Controls:
  w/s : speed + / -
  a/d : steer left / right
  space: stop (speed=0, steer=0)
  x   : re-center steering (delta=0)
  t   : switch to TELEOP mode (keyboard commands drive the car)
  p   : switch to AUTO mode   (forward /path_control to ackermann_cmd)
  q   : quit

Safety:
- On switching mode, we publish STOP for a short duration to ensure
  clean handoff between command sources.
"""


class TerminalTeleop(Node):
  def __init__(self):
    super().__init__("keyboard_teleop_terminal")

    self.declare_parameter("car_name", "utk_car")
    self.declare_parameter("speed", 0.05)  # base speed step (m/s)
    self.declare_parameter("max_speed", 2.0)  # clamp (m/s)
    self.declare_parameter("steering_step", 0.02)  # rad per keypress
    self.declare_parameter("max_steering_angle", 0.34)  # clamp (rad)
    self.declare_parameter("publish_rate_hz", 20.0)

    # Topics for mux
    self.declare_parameter("auto_topic", "path_control")  # from controller

    # Stop-on-switch behavior
    self.declare_parameter("stop_on_switch", True)
    self.declare_parameter("stop_duration_sec", 0.25)  # publish stop this long after switching

    self.car_name = (self.get_parameter("car_name").value or "car").strip("/")
    self.speed_step = float(self.get_parameter("speed").value)
    self.max_speed = float(self.get_parameter("max_speed").value)
    self.steer_step = float(self.get_parameter("steering_step").value)
    self.max_steer = float(self.get_parameter("max_steering_angle").value)

    rate_hz = float(self.get_parameter("publish_rate_hz").value)
    self.period = 1.0 / max(rate_hz, 1e-6)

    self.auto_topic = str(self.get_parameter("auto_topic").value)

    self.stop_on_switch = bool(self.get_parameter("stop_on_switch").value)
    self.stop_duration = float(self.get_parameter("stop_duration_sec").value)

    # Output topic
    self.out_topic = f"/{self.car_name}/ackermann_cmd"
    self.pub = self.create_publisher(AckermannDriveStamped, self.out_topic, 10)

    # Subscribe to AUTO command stream (controller)
    self.auto_sub = self.create_subscription(
        AckermannDriveStamped,
        self.auto_topic,
        self.auto_cb,
        10,
    )

    # Internal teleop state (keyboard)
    self.v = 0.0
    self.delta = 0.0

    # Latest AUTO command received
    self.last_auto_msg = None  # type: AckermannDriveStamped | None

    # Mode: "teleop" or "auto"
    self.mode = "teleop"

    # Latch: after switching mode, publish STOP until this time
    self._stop_until = None  # type: rclpy.time.Time | None

    self.get_logger().info(f"Publishing to: {self.out_topic}")
    self.get_logger().info(f"Subscribing AUTO from: {self.auto_topic}")
    print(HELP)
    self._print_mode_line()

    self.timer = self.create_timer(self.period, self.publish_cb)

    # Setup terminal raw mode
    self._stdin_fd = sys.stdin.fileno()
    self._old_term_settings = termios.tcgetattr(self._stdin_fd)
    tty.setcbreak(self._stdin_fd)

  def destroy_node(self):
    try:
      termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_term_settings)
    except Exception:
      pass
    super().destroy_node()

  def _get_key(self):
    dr, _, _ = select.select([sys.stdin], [], [], 0.0)
    if dr:
      return sys.stdin.read(1)
    return None

  def _clamp(self, x, lo, hi):
    return max(lo, min(hi, x))

  def _print_mode_line(self):
    sys.stdout.write(f"\n[MUX] mode = {self.mode.upper()}   (t=TELEOP, p=AUTO)\n")
    sys.stdout.flush()

  def _stop_now_and_latch(self):
    """Publish a STOP immediately and (optionally) for a short duration."""
    # Reset teleop state so switching back doesn't "jump"
    self.v = 0.0
    self.delta = 0.0

    # Publish one immediate STOP
    stop_msg = AckermannDriveStamped()
    stop_msg.drive.speed = 0.0
    stop_msg.drive.steering_angle = 0.0
    self.pub.publish(stop_msg)

    if self.stop_on_switch and self.stop_duration > 0.0:
      self._stop_until = self.get_clock().now() + Duration(seconds=self.stop_duration)
    else:
      self._stop_until = None

  def auto_cb(self, msg: AckermannDriveStamped):
    self.last_auto_msg = msg

  def _make_teleop_msg(self) -> AckermannDriveStamped:
    msg = AckermannDriveStamped()
    msg.drive.speed = float(self.v)
    msg.drive.steering_angle = float(self.delta)
    return msg

  def _should_publish_stop(self) -> bool:
    if self._stop_until is None:
      return False
    return self.get_clock().now() < self._stop_until

  def publish_cb(self):
    key = self._get_key()

    if key is not None:
      key = key.lower()

      if key == "q":
        self.get_logger().info("Quit.")
        rclpy.shutdown()
        return

      # Mode switching (keyboard mux) + STOP-on-switch
      if key == "t" and self.mode != "teleop":
        self.mode = "teleop"
        self._stop_now_and_latch()
        self._print_mode_line()

      elif key == "p" and self.mode != "auto":
        self.mode = "auto"
        self._stop_now_and_latch()
        self._print_mode_line()

      # TELEOP commands only apply in teleop mode
      elif self.mode == "teleop":
        if key == " ":
          self.v = 0.0
          self.delta = 0.0
        elif key == "w":
          self.v = self._clamp(self.v + self.speed_step, -self.max_speed, self.max_speed)
        elif key == "s":
          self.v = self._clamp(self.v - self.speed_step, -self.max_speed, self.max_speed)
        elif key == "a":
          self.delta = self._clamp(self.delta + self.steer_step, -self.max_steer, self.max_steer)
        elif key == "d":
          self.delta = self._clamp(self.delta - self.steer_step, -self.max_steer, self.max_steer)
        elif key == "x":
          self.delta = 0.0

        sys.stdout.write(f"\r mode={self.mode:<6}  speed={self.v:+.2f} m/s   steer={self.delta:+.3f} rad   ")
        sys.stdout.flush()
      else:
        # AUTO mode: ignore w/a/s/d/space/x to avoid accidental override
        sys.stdout.write(f"\r mode={self.mode:<6}  (forwarding '{self.auto_topic}' -> '{self.out_topic}')   ")
        sys.stdout.flush()

    # If we recently switched modes, publish STOP for a short duration
    if self._should_publish_stop():
      stop_msg = AckermannDriveStamped()
      stop_msg.drive.speed = 0.0
      stop_msg.drive.steering_angle = 0.0
      self.pub.publish(stop_msg)
      return

    # Publish based on current mode
    if self.mode == "teleop":
      self.pub.publish(self._make_teleop_msg())
    else:
      if self.last_auto_msg is not None:
        self.pub.publish(self.last_auto_msg)
      # else: nothing yet


def main(args=None):
  rclpy.init(args=args)
  node = TerminalTeleop()
  try:
    rclpy.spin(node)
  finally:
    if rclpy.ok():
      rclpy.shutdown()
    node.destroy_node()


if __name__ == "__main__":
  main()
