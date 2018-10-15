#include "img_server.hpp"



//------------------------------Image Server's Functions------------------------------
ImgServer::ImgServer(int size) {
    buffer.resize(size);
}


bool ImgServer::accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res) {
    res.rgbd_image.rgb_image = *(buffer[(req.frame)%150].first);
    res.rgbd_image.depth_image = *(buffer[(req.frame)%150].second);
    return true;
}


void ImgServer::camCallbackRGB(const sensor_msgs::Image::ConstPtr& img) {
    buffer[(img->header.seq)%150].first = img;
}


void ImgServer::camCallbackDepth(const sensor_msgs::Image::ConstPtr& img) {
    buffer[(img->header.seq)%150].second = img;
}
//------------------------------Image Server's Functions------------------------------