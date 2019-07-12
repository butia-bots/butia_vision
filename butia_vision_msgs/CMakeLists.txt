cmake_minimum_required(VERSION 2.8.3)
project(butia_vision_msgs)

find_package(catkin REQUIRED COMPONENTS std_msgs sensor_msgs genmsg message_generation)

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
  ListClasses.srv
  PeopleIntroducing.srv
  SegmentationRequest.srv
  StartTracking.srv
  StopTracking.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(CATKIN_DEPENDS sensor_msgs std_msgs message_runtime)

include_directories()