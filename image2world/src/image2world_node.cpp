#include "ros/ros.h"

#include "image2world/image2world.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image2world_node");

    ros::NodeHandle nh;

    Image2World image2world(nh);

    ros::spin();
    return 0;
}