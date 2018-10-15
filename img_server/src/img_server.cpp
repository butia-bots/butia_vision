#include "img_server.hpp"



//------------------------------Image Prepare's Functions------------------------------
ImgServer::ImgServer(ros::NodeHandle &nh) : node_handle(nh), min_seq(0), max_seq(0), last_rgb(-1), buffer_size(150) {
    rgb_buffer.resize(buffer_size);
    depth_buffer.resize(buffer_size);

    img_rgb_sub = nh.subscribe("/kinect2/qhd/image_color_rect", 100, &ImgServer::camCallBackRGB, this);
    img_d_sub = nh.subscribe("/kinect2/qhd/image_depth_rect", 100, &ImgServer::camCallBackDepth, this);
    service = nh.advertiseService("/vision_system/is/image_request", &ImgServer::accessQueue, this);
}


ImgServer::ImgServer(ros::NodeHandle &nh, int size) : node_handle(nh), min_seq(0), max_seq(0), last_rgb(-1), buffer_size(size) {
    rgb_buffer.resize(buffer_size);
    depth_buffer.resize(buffer_size);

    img_rgb_sub = nh.subscribe("/kinect2/qhd/image_color_rect", 100, &ImgServer::camCallBackRGB, this);
    img_d_sub = nh.subscribe("/kinect2/qhd/image_depth_rect", 100, &ImgServer::camCallBackDepth, this);
    service = nh.advertiseService("/vision_system/is/image_request", &ImgServer::accessQueue, this);
}

bool ImgServer::accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res) {
    int seq = req.frame;
    if(seq > max_seq || seq < min_seq) return false;
    res.rgbd_image.rgb = *(rgb_buffer[seq%buffer_size]);
    
    if(depth_buffer[seq%buffer_size] == NULL) return false;
    res.rgbd_image.depth = *(depth_buffer[seq%buffer_size]);
    
    return true;
}

void ImgServer::camCallBackRGB(const sensor_msgs::Image::ConstPtr img) {
    int seq = img->header.seq;
    ROS_INFO("RGB Frame ID: %d", seq);
    rgb_buffer[seq%buffer_size] = img;
    depth_buffer[seq%buffer_size] = NULL;
    if(seq - buffer_size >= min_seq) min_seq = seq;
    max_seq = seq;
    last_rgb = seq%buffer_size;
}

void ImgServer::camCallBackDepth(const sensor_msgs::Image::ConstPtr img) {
    ROS_INFO("Depth Frame ID: %d", img->header.seq);
    if(last_rgb >= 0) depth_buffer[last_rgb] = img;
    last_rgb = -1;
}
//------------------------------Image Prepare's Functions------------------------------