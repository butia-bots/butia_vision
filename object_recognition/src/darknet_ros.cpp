#include "darknet_ros.h"

DarknetROS::DarknetROS(ros::NodeHandle _nh) : nh(_nh)
{
    loadParameters();

    bounding_boxes_sub = nh.subscribe();
}

void DarknetROS::loadParameters()
{
    
}
