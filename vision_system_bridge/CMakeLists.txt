cmake_minimum_required(VERSION 2.8.3)
project(vision_system_bridge)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs std_srvs cv_bridge vision_system_msgs)
find_package(OpenCV REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES vision_system_bridge
    CATKIN_DEPENDS roscpp std_msgs std_srvs sensor_msgs cv_bridge vision_system_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/vision_system_bridge.cpp)

add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PROJECT_NAME})
