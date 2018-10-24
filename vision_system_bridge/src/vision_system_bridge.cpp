#include "vision_system_bridge.h"

VisionSystemBridge::VisionSystemBridge(ros::NodeHandle &nh) : node_handle(nh), min_seq(0), max_seq(0), last_rgb(-1), last_depth(-1)
{
    readParameters();
    
    resizeBuffers();

    image_rgb_sub = new image_transport::SubscriberFilter(it, image_rgb_topic, queue_size);
    image_depth_sub = new image_transport::SubscriberFilter(it, image_depth_topic, queue_size);
    camera_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(node_handle, camera_info_topic, queue_size);

    if(use_exact_time) {
        exact_sync = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(queue_size), *image_rgb_sub, *image_depth_sub, *camera_info_sub);
        exact_sync->registerCallback(boost::bind(&VisionSystemBridge::kinectCallback, this, _1, _2, _3, _4));
    }
    else {
        approximate_sync = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(queue_size),  *image_rgb_sub, *image_depth_sub, *camera_info_sub);
        approximate_sync->registerCallback(boost::bind(&VisionSystemBridge::kinectCallback, this, _1, _2, _3, _4));
    }

}

void VisionSystemBridge::resizeBuffers()
{
    image_rgb_buffer.resize(buffer_size);
    image_depth_buffer.resize(buffer_size);
    camera_info_buffer.resize(buffer_size);
}


void VisionSystemBridge::readParameters()
{
    node_handle.param("/vision_system_bridge/subscribers/queue_size", queue_size, 5);

    node_handle.param("/vision_system_bridge/subscribers/image_rgb/topic", image_rgb_topic, std::string("/kinect2/qhd/image_color_rect"));

    node_handle.param("/vision_system_bridge/subscribers/image_depth/topic", image_depth_topic, std::string("/kinect2/qhd/image_depth_rect"));

    node_handle.param("/vision_system_bridge/subscribers/camera_info/topic", camera_info_topic, std::string("/kinect2/qhd/camera_info"));

    node_handle.param("/vision_system_bridge/services/rgbdimage_request/service", rgbdimage_request_service, std::string("/vision_system/vsb/rgbdimage_request"));

    node_handle.param("/vision_system_bridge/buffer_size", buffer_size, 150);
    node_handle.param("/vision_system_bridge/use_exact_time", use_exact_time, false);
}

void VisionSystemBridge::kinectCallback(sensor_msgs::Image::ConstPtr image_rgb, sensor_msgs::Image::ConstPtr image_depth, sensor_msgs::CameraInfo::ConstPtr camera_info)
{
    int seq = image_rgb->header.seq;
    ROS_INFO("RGB Frame ID: %d", seq);
    ROS_INFO("DEPTH Frame ID: %d", image_depth);
    /*rgb_buffer[seq%buffer_size] = img;

    if(last_depth > last_rgb) {
        last_rgb = (last_rgb/buffer_size)*buffer_size +  seq%buffer_size;
        depth_buffer[last_rgb%buffer_size] = depth_buffer[last_depth%buffer_size];
        last_depth = last_rgb;
    }
    else {
        last_rgb = (last_rgb/buffer_size)*buffer_size +  seq%buffer_size;
    }

    if(seq - buffer_size >= min_seq || seq < min_seq) min_seq = seq;
    max_seq = seq;*/

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
