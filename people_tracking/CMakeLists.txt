cmake_minimum_required(VERSION 2.8.3)
project(people_tracking)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_node src/people_tracking_node.cpp)

target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})