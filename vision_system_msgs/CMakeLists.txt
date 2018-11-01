cmake_minimum_required(VERSION 2.8.3)
project(vision_system_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs genmsg message_generation std_srvs)

add_message_files(
  FILES
  BoundingBox.msg
  Description.msg
  Description3D.msg
  Recognitions.msg
  Recognitions3D.msg
  ClassifierReload.msg
  RGBDImage.msg
)

add_service_files(
  FILES
  FaceClassifierTraining.srv
  ImageRequest.srv
  PeopleIntroducing.srv
  SegmentationRequest.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  std_srvs
)

catkin_package(CATKIN_DEPENDS message_runtime sensor_msgs std_srvs)

include_directories()