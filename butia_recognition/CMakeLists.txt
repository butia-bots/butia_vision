cmake_minimum_required(VERSION 2.8.3)
project(butia_recognition)

find_package(catkin REQUIRED rospy std_msgs std_srvs sensor_msgs geometry_msgs butia_vision_msgs)

catkin_python_setup()

catkin_package(
    INCLUDE_DIRS
    LIBRARIES 
    CATKIN_DEPENDS std_msgs std_srvs sensor_msgs butia_vision_msgs
)

install(PROGRAMS
  scripts/butia_recognition/yolov5_recognition.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)