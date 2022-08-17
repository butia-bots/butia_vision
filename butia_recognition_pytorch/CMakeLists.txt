cmake_minimum_required(VERSION 2.8.3)
project(butia_recognition_pytorch)

find_package(catkin REQUIRED rospy std_msgs std_srvs sensor_msgs butia_vision_msgs)

install(PROGRAMS
  scripts/butia_recognition_pytorch_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)