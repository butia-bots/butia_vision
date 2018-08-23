cmake_minimum_required(VERSION 2.8.3)
project(img_server)

find_package(catkin REQUIRED COMPONENTS roscpp image_transport std_msgs sensor_msgs std_srvs camera_info_manager cv_bridge)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

catkin_package(
    CATKIN_DEPENDS roscpp image_transport std_msgs std_srvs sensor_msgs camera_info_manager cv_bridge
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_executable(img_server_node src/img_server_node.cpp)

target_link_libraries(img_server_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})