#!/usr/bin/env python3

"""Publish the nth Fibonacci number."""

import time

import rclpy  # ROS2 Python API
from rclpy.node import Node
from std_msgs.msg import Int64, String  # ROS message type


# You may combine compute_fibonacci in this file directly.
def compute_fibonacci(n):
    """Return the nth Fibonacci number."""
    # BEGIN QUESTION 1.1

    # END QUESTION 1.1


def main(args=None):

    # NOTE (ROS2): When using console_scripts, ROS2 expects a function named
    # "main" to exist, because setup.py points to proj1.fibonacci:main.

    # You can uncomment the following lines after completing compute_fibonacci.
    # fibonacci_numbers = [compute_fibonacci(i) for i in range(11)]
    # print(fibonacci_numbers)

    # In ROS2, we must initialize rclpy before creating nodes.
    rclpy.init(args=args)

    # Every node registers itself with the ROS graph, so we must pass a
    # name for the node. (A good practice is to use the same name as the script,
    # but it's up to you.)
    node = Node("fibonacci")

    # Here's an example to get parameters from ROS2.
    # In ROS2, parameters must be declared before they are retrieved.
    #
    # Function signature:
    #   node.declare_parameter(parameter_name, default_value)
    #   node.get_parameter(parameter_name).value
    node.declare_parameter("output_topic", "/proj1/fib_output")
    node.declare_parameter("index", 10)

    fib_topic = node.get_parameter("output_topic").value

    # Get the value of the "index" parameter from ROS.
    # Don't forget to make sure it's the right type (an integer)!
    # Feel free to specify a default value above.
    # BEGIN QUESTION 1.2

    # END QUESTION 1.2

    # Here's an example to create a ROS publisher. The queue_size defines how
    # many messages to buffer when waiting to send messages. In general, a large
    # number will be useful when many messages will be sent. We are sending
    # messages infrequently, 1 is okay.
    #
    # Function signature:
    #   node.create_publisher(msg_type, topic_name, queue_size)
    example_publisher = node.create_publisher(
        String,
        "/proj1/example",
        1,
    )

    # Using the above example as a guide, create a publisher to the topic name
    # you obtained earlier. The example publisher publishes String messages;
    # what type of messages do we want to publish from our Fibonacci publisher?
    # (Hint: check the imports at the top of the file.)
    # BEGIN QUESTION 1.2

    # END QUESTION 1.2

    # It can take some time for the publisher to be ready, so we'll wait before
    # we start to publish. (The specific wait time isn't important, but for
    # completeness, this waits for 0.5 seconds.)
    # time.sleep(2.0)

    example_publisher.publish(String(data="Hello World!"))

    # Call compute_fibonacci with the index you obtained earlier, then publish
    # the resulting Fibonacci number.
    # BEGIN QUESTION 1.2

    # END QUESTION 1.2

    node.get_logger().info(f"Published Fibonacci({index}) = {result}")

    # Spin briefly so the message gets sent before shutdown.
    rclpy.spin_once(node, timeout_sec=1.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
