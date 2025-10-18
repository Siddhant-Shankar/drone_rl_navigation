cmake_minimum_required(VERSION 3.0.2)
project(drone_rl_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  DroneState.msg
  PathPoint.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs nav_msgs visualization_msgs sensor_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/rl_controller.py
  scripts/path_visualizer.py
  scripts/drone_simulator.py
  scripts/training_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
