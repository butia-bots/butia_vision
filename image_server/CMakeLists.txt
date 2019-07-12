cmake_minimum_required(VERSION 2.8.3)
project(image_server)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  message_filters
  roscpp
  sensor_msgs
  butia_vision_msgs
)

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES image_server
  CATKIN_DEPENDS image_transport message_filters roscpp sensor_msgs butia_vision_msgs
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/image_server.cpp
)

add_executable(${PROJECT_NAME}_node src/image_server_node.cpp)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)