cmake_minimum_required(VERSION 2.8.3)
project(shelf_detection)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED rospy cv_bridge butia_vision_msgs std_msgs sensor_msgs)

find_package(OpenCV REQUIRED PATHS "/usr/local/lib")

catkin_package(
  LIBRARIES shelf_detection
  CATKIN_DEPENDS rospy cv_bridge butia_vision_msgs std_msgs sensor_msgs
)

include_directories(
# include
  ${OpenCV_INCLUDE_DIRS}
# ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/shelf_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

