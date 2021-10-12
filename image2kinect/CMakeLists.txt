cmake_minimum_required(VERSION 2.8.3)
project(image2kinect)

add_compile_options(-std=c++14)

find_package(PCL 1.10 REQUIRED COMPONENTS io registration segmentation visualization)
find_package(catkin REQUIRED roscpp butia_vision_msgs geometry_msgs cv_bridge tf pcl_ros)
find_package(OpenCV REQUIRED PATHS "/usr/local/lib")

catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES image2kinect
  CATKIN_DEPENDS roscpp butia_vision_msgs geometry_msgs cv_bridge
)


include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
 ${PCL_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(${PROJECT_NAME}
  src/image2kinect.cpp
)


add_executable(${PROJECT_NAME}_node src/image2kinect_node.cpp)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROJECT_NAME}
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  ${PROJECT_NAME}
)
