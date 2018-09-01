#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "sensor_msgs/Image.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/ImageRequest.h"



std::vector<sensor_msgs::ImageConstPtr> buffer;


bool accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res) {
    res.image = *(buffer[(req.frame)%150]);
    return true;
}


void camCallBack(const sensor_msgs::ImageConstPtr img) {
    buffer[(img->header.seq)%150] = img;
}


int main(int argc, char **argv) {
    buffer.resize(150);
    ros::init(argc, argv, "img_server_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber img_sub = it.subscribe("usb_cam/image_raw", 100, &camCallBack);
    ros::ServiceServer service = nh.advertiseService("/vision_system/img_server/image_request", &accessQueue);
    ros::spin();
}