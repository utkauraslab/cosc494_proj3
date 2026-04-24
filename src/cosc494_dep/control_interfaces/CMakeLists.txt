cmake_minimum_required(VERSION 3.8)
project(control_interfaces)

# Core build system
find_package(ament_cmake REQUIRED)

# Interface generation
find_package(rosidl_default_generators REQUIRED)

# These must be found BEFORE using them in rosidl_generate_interfaces()
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# Generate services
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/FollowPath.srv"
  DEPENDENCIES builtin_interfaces std_msgs nav_msgs
)

# Let downstream packages use the generated typesupport
ament_export_dependencies(rosidl_default_runtime)

ament_package()