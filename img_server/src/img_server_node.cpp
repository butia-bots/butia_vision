#include <vector>
#include <ros/ros.h>
#include <string>
#include <opencv2/opencv.hpp>

#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"


std::vector<sensor_msgs::Image> img_server;


void camCallback(const sensor_msgs::Image img) {
    img_server[(img.header.seq)%(img_server.size())] = img;
}

int main(int argc, char **argv) {
    int i = 0;
    cv_bridge::CvImagePtr img2show;

    img_server.resize(150);

    ros::init(argc, argv, "img_server");
    ros::NodeHandle nh;
    ros::Subscriber img_sub = nh.subscribe("usb_cam/image_raw", 150, camCallback);

    cv::namedWindow("Images of the Queue", cv::WINDOW_AUTOSIZE);

    for (;;) {
        img2show = cv_bridge::toCvCopy(img_server[i], img_server[i].encoding);
        cv::imshow("Images of the Queue", img2show->image);
        if (i == 149)
            i = 0;
        else
            i++;
    }
}