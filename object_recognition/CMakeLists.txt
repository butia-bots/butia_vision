cmake_minimum_required(VERSION 2.8.3)
project(object_recognition)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED roscpp darknet_ros_msgs butia_vision_msgs geometry_msgs cv_bridge)

find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES object_recognition
  CATKIN_DEPENDS roscpp butia_vision_msgs darknet_ros_msgs geometry_msgs cv_bridge
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/yolo_recognition.cpp
)


add_executable(${PROJECT_NAME}_node src/object_recognition_node.cpp)

install(PROGRAMS
  scripts/people_detection_node.py
  scripts/object_recognition_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)
