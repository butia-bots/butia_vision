cmake_minimum_required(VERSION 2.8.3)
project(object_recognition)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED roscpp darknet_ros_msgs vision_system_msgs geometry_msgs)

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES object_recognition
  CATKIN_DEPENDS roscpp vision_system_msgs darknet_ros_msgs geometry_msgs
)

include_directories(
 include
 ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/yolo_recognition.cpp
)


add_executable(${PROJECT_NAME}_node src/object_recognition_node.cpp)


target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)
