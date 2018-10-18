cmake_minimum_required(VERSION 2.8.3)
project(image2world)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED roscpp vision_system_msgs geometry_msgs cv_bridge)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES image2world
  CATKIN_DEPENDS roscpp vision_system_msgs geometry_msgs cv_bridge
)


include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/image2world.cpp
)


add_executable(${PROJECT_NAME}_node src/image2world_node.cpp)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}
)
