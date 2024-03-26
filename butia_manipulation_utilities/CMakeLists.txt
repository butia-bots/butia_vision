cmake_minimum_required(VERSION 2.8.3)
project(butia_grasp_generator)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED rospy butia_vision_msgs std_msgs sensor_msgs)

find_package(OpenCV REQUIRED)

catkin_package(
  LIBRARIES butia_grasp_generator
  CATKIN_DEPENDS rospy butia_vision_msgs std_msgs sensor_msgs
)

include_directories(
# include
  ${OpenCV_INCLUDE_DIRS}
# ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/grasp_generator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
