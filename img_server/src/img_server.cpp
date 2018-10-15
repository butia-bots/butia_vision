#include "img_server.hpp"



//------------------------------Image Prepare's Functions------------------------------
ImgServer::ImgServer() {
    buffer.resize(150);
}


ImgServer::ImgServer(int size) {
    buffer.resize(size);
}


std::vector<std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>> ImgServer::getBuffer() {
    return buffer;
}


bool ImgServer::accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res) {
    res.rgbd_image.rgb = *(buffer[(req.frame)%150].first);
    res.rgbd_image.depth = *(buffer[(req.frame)%150].second);
    return true;
}


void ImgServer::camCallBackRGB(const sensor_msgs::ImageConstPtr img) {
    ROS_INFO("RGB Frame ID: %d", img->header.seq);
    buffer[(img->header.seq)%150].first = img;
}


void ImgServer::camCallBackDepth(const sensor_msgs::ImageConstPtr img) {
    ROS_INFO("Depth Frame ID: %d", img->header.seq);
    buffer[(img->header.seq)%150].second = img;
}
//------------------------------Image Prepare's Functions------------------------------