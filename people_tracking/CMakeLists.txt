cmake_minimum_required(VERSION 2.8.3)
project(people_tracking)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED rospy cv_bridge butia_vision_msgs std_msgs sensor_msgs)

find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES people_tracking
  CATKIN_DEPENDS rospy cv_bridge butia_vision_msgs std_msgs sensor_msgs
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/yolo_recognition.cpp
)

add_executable(${PROJECT_NAME}_node src/people_tracking_node.cpp)

install(PROGRAMS
  scripts/people_tracking_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

