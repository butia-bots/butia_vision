cmake_minimum_required(VERSION 2.8.3)
project(doris_cam)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp rospy)
find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
    INCLUDE_DIRS include
)

add_executable(${PROJECT_NAME}_node src/doris_cam_node.cpp)