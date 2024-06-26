cmake_minimum_required(VERSION 2.8.3)
project(butia_manipulation_utilities)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED rospy butia_vision_msgs std_msgs sensor_msgs tf2 tf2_ros tf2_msgs tf2_sensor_msgs)

find_package(OpenCV REQUIRED)

catkin_package(
  LIBRARIES butia_manipulation_utilities
  CATKIN_DEPENDS rospy butia_vision_msgs std_msgs sensor_msgs tf2 tf2_ros tf2_msgs tf2_sensor_msgs
)

include_directories(
# include
  ${OpenCV_INCLUDE_DIRS}
# ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/grasp_generator/grasp_generator_node.py
  scripts/placeable_area_generator/placeable_area_generator_node.py
  scripts/placeable_area_generator/call_service.py
  scripts/placeable_area_generator/shelf_call_service.py
  scripts/placeable_area_generator/shelf_classification_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
