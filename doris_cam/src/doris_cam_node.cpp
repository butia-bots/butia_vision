#include <opencv2/opencv.hpp>
#include <queue>
#include <sstream>
#include <ros/ros.h>

#include "doris_cam/includes/cam_config.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "cam_pub");
    ros::NodeHandle nh;

    ros::Publisher img_pub = nh.advertise<vision_system_msgs::ImgRaw>("img_raw", 1000);
}