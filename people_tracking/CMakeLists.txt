cmake_minimum_required(VERSION 2.8.3)
project(people_tracking)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp vision_system_msgs img_server)
find_package(OpenCV REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp vision_system_msgs img_server
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME}_node src/people_tracking_node.cpp)

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})