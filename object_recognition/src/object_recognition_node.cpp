#include <ros.h>

#include "darknet_ros.h"

#include "darknet_ros_msgs/BoundingBoxes.h"
#include "vision_system_msgs/RecognizedObjects.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "object_recognition_node")

    ros::NodeHandle nh;

    DarknetROS object_recognition(nh);
}
