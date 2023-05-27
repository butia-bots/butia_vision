#include "ros/ros.h"

#include "image2kinect/image2kinect.h"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image2kinect_node");

    ros::NodeHandle nh;

    Image2Kinect image2kinect(nh);

    ros::spin();
    return 0;
}