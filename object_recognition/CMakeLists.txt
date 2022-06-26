cmake_minimum_required(VERSION 2.8.3)
project(object_recognition)

find_package(catkin REQUIRED rospy std_msgs std_srvs sensor_msgs butia_vision_msgs cv_bridge)

find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
)

install(PROGRAMS
  scripts/people_detection_node.py
  scripts/object_recognition_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)