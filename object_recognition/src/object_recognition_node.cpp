#include "ros/ros.h"

#include "object_recognition/yolo_recognition.h"


int main(int argc, char **argv)
{
    ros::init(argc, argv, "object_recognition_node");

    ros::NodeHandle nh;

    YoloRecognition object_recognition(nh);

    ros::spin();
    return 0;
}
