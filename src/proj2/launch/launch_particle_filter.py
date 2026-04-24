from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    # ---- Resolve default parameter file path ----
    proj2_pkg_share = get_package_share_directory("proj2")
    default_param_file = os.path.join(proj2_pkg_share, "config", "parameters_ros2.yaml")

    # ---- Launch Arguments ----
    publish_tf_arg = DeclareLaunchArgument("publish_tf", default_value="true", description="Whether to publish TF")
    tf_prefix_arg = DeclareLaunchArgument("tf_prefix", default_value="", description="TF frame prefix")
    n_particles_arg = DeclareLaunchArgument("n_particles", default_value="10000", description="Number of particles")
    n_viz_particles_arg = DeclareLaunchArgument("n_viz_particles", default_value="60", description="Number of particles to visualize")
    scan_topic_arg = DeclareLaunchArgument("scan_topic", default_value="scan", description="Laser scan topic")
    laser_ray_step_arg = DeclareLaunchArgument("laser_ray_step", default_value="18", description="Step size for downsampling laser rays")
    exclude_max_range_rays_arg = DeclareLaunchArgument("exclude_max_range_rays", default_value="true", description="Whether to exclude max-range rays")
    max_range_meters_arg = DeclareLaunchArgument("max_range_meters", default_value="11.0", description="Maximum laser range in meters")
    initial_x_arg = DeclareLaunchArgument("initial_x", default_value="NaN", description="Initial x position")
    initial_y_arg = DeclareLaunchArgument("initial_y", default_value="NaN", description="Initial y position")
    initial_theta_arg = DeclareLaunchArgument("initial_theta", default_value="NaN", description="Initial heading")
    param_file_arg = DeclareLaunchArgument("param_file", default_value=default_param_file, description="Path to parameters.yaml")

    publish_tf = LaunchConfiguration("publish_tf")
    tf_prefix = LaunchConfiguration("tf_prefix")
    n_particles = LaunchConfiguration("n_particles")
    n_viz_particles = LaunchConfiguration("n_viz_particles")
    scan_topic = LaunchConfiguration("scan_topic")
    laser_ray_step = LaunchConfiguration("laser_ray_step")
    exclude_max_range_rays = LaunchConfiguration("exclude_max_range_rays")
    max_range_meters = LaunchConfiguration("max_range_meters")
    initial_x = LaunchConfiguration("initial_x")
    initial_y = LaunchConfiguration("initial_y")
    initial_theta = LaunchConfiguration("initial_theta")
    param_file = LaunchConfiguration("param_file")

    # ---- particle_filter node ----
    particle_filter_node = Node(
        package="proj2",
        executable="particle_filter",
        name="particle_filter",
        output="screen",
        parameters=[
            {
                "publish_tf": publish_tf,
                "tf_prefix": tf_prefix,
                "n_particles": n_particles,
                "n_viz_particles": n_viz_particles,
                "scan_topic": scan_topic,
                "laser_ray_step": laser_ray_step,
                "exclude_max_range_rays": exclude_max_range_rays,
                "max_range_meters": max_range_meters,
                "initial_x": initial_x,
                "initial_y": initial_y,
                "initial_theta": initial_theta,
            },
            param_file,
        ],
    )

    return LaunchDescription(
        [
            publish_tf_arg,
            tf_prefix_arg,
            n_particles_arg,
            n_viz_particles_arg,
            scan_topic_arg,
            laser_ray_step_arg,
            exclude_max_range_rays_arg,
            max_range_meters_arg,
            initial_x_arg,
            initial_y_arg,
            initial_theta_arg,
            param_file_arg,
            particle_filter_node,
        ]
    )
