#include "darknet_ros.h"

DarknetROS::DarknetROS(ros::NodeHandle _nh) : nh(_nh)
{
    setParameters();

    bounding_boxes_sub = ros::Subscriber()
}