cmake_minimum_required(VERSION 2.8.3)
project(instance_segmentation)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED roscpp rospy mask_rcnn_ros butia_vision_msgs geometry_msgs cv_bridge)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy butia_vision_msgs mask_rcnn_ros geometry_msgs cv_bridge
)

install(PROGRAMS
  scripts/instance_segmentation_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
