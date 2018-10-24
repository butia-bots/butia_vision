#include "vision_system_bridge.h"

VisionSystemBridge::VisionSystemBridge(ros::NodeHandle &nh) : node_handle(nh), min_seq(0), max_seq(0), last_rgb(-1), last_depth(-1), buffer_size(150) {
    rgb_buffer.resize(buffer_size);
    depth_buffer.resize(buffer_size);

    img_rgb_sub = nh.subscribe("/kinect2/qhd/image_color_rect", 100, &VisionSystemBridge::camCallBackRGB, this);
    img_d_sub = nh.subscribe("/kinect2/qhd/image_depth_rect", 100, &VisionSystemBridge::camCallBackDepth, this);
    service = nh.advertiseService("/vision_system/is/image_request", &VisionSystemBridge::accessQueue, this);
}


void VisionSystemBridge::readParameters() {
    node_handle.param("/vision_system_bridge/subscribers/image_rgb/topic", image_rgb_topic, std::string("/kinect2/qhd/image_color_rect"));
    node_handle.param("/vision_system_bridge/subscribers/image_rgb/queue_size", image_rgb_qs, 100);

    node_handle.param("/vision_system_bridge/subscribers/image_depth/topic", image_depth_topic, std::string("/kinect2/qhd/image_depth_rect"));
    node_handle.param("/vision_system_bridge/subscribers/image_depth/queue_size", image_depth_qs, 100);

    node_handle.param("/vision_system_bridge/services/rgbdimage_request/service", rgbdimage_request_service, std::string("/vision_system/vsb/rgbdimage_request"));
}


/*
bool VisionSystemBridge::accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res) {
    int seq = req.frame;
    if(seq > max_seq || seq < min_seq) 
    {
        ROS_ERROR("INVALID RGB ACCESS!");
        return false;
    }
    res.rgbd_image.rgb = *(rgb_buffer[seq%buffer_size]);
    
    if(seq%buffer_size > last_depth%buffer_size) {
        ROS_ERROR("INVALID DEPTH ACCESS!");
        return false;
    }
    res.rgbd_image.depth = *(depth_buffer[seq%buffer_size]);
    
    return true;
}

void VisionSystemBridge::camCallBackRGB(const sensor_msgs::Image::ConstPtr img) {
    int seq = img->header.seq;
    ROS_INFO("RGB Frame ID: %d", seq);
    rgb_buffer[seq%buffer_size] = img;

    if(last_depth > last_rgb) {
        last_rgb = (last_rgb/buffer_size)*buffer_size +  seq%buffer_size;
        depth_buffer[last_rgb%buffer_size] = depth_buffer[last_depth%buffer_size];
        last_depth = last_rgb;
    }
    else {
        last_rgb = (last_rgb/buffer_size)*buffer_size +  seq%buffer_size;
    }

    if(seq - buffer_size >= min_seq || seq < min_seq) min_seq = seq;
    max_seq = seq;
}

void VisionSystemBridge::camCallBackDepth(const sensor_msgs::Image::ConstPtr img) {
    ROS_INFO("Depth Frame ID: %d", img->header.seq);
    if(last_rgb == last_depth) depth_buffer[(++last_depth)%buffer_size] = img;
    else if(last_rgb > last_depth) {
        last_depth = last_rgb;
        depth_buffer[last_depth%buffer_size] = img;
    }
}*/
