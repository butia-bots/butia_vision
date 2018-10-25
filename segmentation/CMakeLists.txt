cmake_minimum_required(VERSION 2.8.3)
project(segmentation)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp vision_system_msgs img_server cv_bridge)
find_package(OpenCV REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp vision_system_msgs img_server cv_bridge
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/${PROJECT_NAME}.cpp)

add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PROJECT_NAME})