cmake_minimum_required(VERSION 2.8.3)
project(butia_vision_bridge)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs std_srvs cv_bridge tf butia_vision_msgs message_filters image_transport)
find_package(OpenCV REQUIRED PATHS "/usr/local/lib")

catkin_package(
    INCLUDE_DIRS include/${PROJECT_NAME}
    LIBRARIES butia_vision_bridge
    CATKIN_DEPENDS roscpp std_msgs std_srvs sensor_msgs cv_bridge butia_vision_msgs message_filters image_transport
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/butia_vision_bridge.cpp)

add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PROJECT_NAME})

install(PROGRAMS
  scripts/butia_vision_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)