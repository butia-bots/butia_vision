cmake_minimum_required(VERSION 2.8.3)
project(vision_system_msgs)

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
  Image2World.srv
  SegmentationRequest.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

include_directories()