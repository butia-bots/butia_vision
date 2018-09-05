#include "ros/ros.h"

#include "yolo_recognition.h"

#include "darknet_ros_msgs/BoundingBoxes.h"
#include "vision_system_msgs/RecognizedObjects.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "object_recognition_node");

    ros::NodeHandle nh;

    YoloRecognition object_recognition(nh);

    ros::spin();
    return 0;
}
