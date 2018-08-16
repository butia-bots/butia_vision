#include <opencv2/opencv.hpp>
#include <ros/ros.h>

#include "sensor_msgs/Image.h"
#include "cam_config.h"


int main(int argc, char **argv) {
    //ros::init(argc, argv, "cam_pub");
    //ros::NodeHandle nh;
    //ros::Publisher img_pub = nh.advertise<sensor_msgs::Image>("img_raw", 1000);

    DorisCam cam_controller;
    cv::namedWindow("Testing Camera Node", 1);
    
    if (cam_controller.startRecording() == EXIT_SUCCESS) {
        ROS_INFO("***STARTING RECORDING***");
        for (;;) {
            cv::Mat frame;
            cam_controller.cap >> frame;
            cv::imshow("Testing Camera Nodes", frame);
            if (cv::waitKey(30) >= 0) {
                if (cam_controller.stopRecording() == EXIT_SUCCESS)
                    ROS_INFO("***RECORDING STOPPED***");
                else
                    ROS_ERROR("***FAILED TO CLOSE CAMERA");
            }
        }
    } else
        ROS_ERROR("***FAILED TO OPEN CAMERA***");

    return 0;
}