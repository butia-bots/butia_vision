cmake_minimum_required(VERSION 2.8.3)
project(people_tracking)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS roscpp vision_system_msgs cv_bridge)
find_package(OpenCV REQUIRED PATHS "/usr/local/lib")

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES people_tracking
    CATKIN_DEPENDS roscpp vision_system_msgs cv_bridge
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/people_tracking.cpp)

add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${PROJECT_NAME})
