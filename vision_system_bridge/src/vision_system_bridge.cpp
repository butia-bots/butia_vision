#include "vision_system_bridge/vision_system_bridge.h"

VisionSystemBridge::VisionSystemBridge(ros::NodeHandle &nh) : node_handle(nh), it(nh), seq(0)
{
    readParameters();
    
    image_rgb_sub = new image_transport::SubscriberFilter(it, image_rgb_sub_topic, sub_queue_size);
    image_depth_sub = new image_transport::SubscriberFilter(it, image_depth_sub_topic, sub_queue_size);
    camera_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(node_handle, camera_info_sub_topic, sub_queue_size);

    if(use_exact_time) {
        exact_sync = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(sub_queue_size), *image_rgb_sub, *image_depth_sub, *camera_info_sub);
        exact_sync->registerCallback(boost::bind(&VisionSystemBridge::kinectCallback, this, _1, _2, _3));
    }
    else {
        approximate_sync = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(sub_queue_size),  *image_rgb_sub, *image_depth_sub, *camera_info_sub);
        approximate_sync->registerCallback(boost::bind(&VisionSystemBridge::kinectCallback, this, _1, _2, _3));
    }

    image_rgb_pub = it.advertise(image_rgb_pub_topic, pub_queue_size);
    image_depth_pub = it.advertise(image_depth_pub_topic, pub_queue_size);
    camera_info_pub = node_handle.advertise<sensor_msgs::CameraInfo>(camera_info_pub_topic, pub_queue_size);
}

void VisionSystemBridge::readParameters()
{
    node_handle.param("/vision_system_bridge/subscribers/queue_size", sub_queue_size, 5);
    node_handle.param("/vision_system_bridge/subscribers/image_rgb/topic", image_rgb_sub_topic, std::string("/kinect2/qhd/image_color_rect"));
    node_handle.param("/vision_system_bridge/subscribers/image_depth/topic", image_depth_sub_topic, std::string("/kinect2/qhd/image_depth_rect"));
    node_handle.param("/vision_system_bridge/subscribers/camera_info/topic", camera_info_sub_topic, std::string("/kinect2/qhd/camera_info"));

    node_handle.param("/vision_system_bridge/publishers/queue_size", pub_queue_size, 5);   
    node_handle.param("/vision_system_bridge/publishers/image_rgb/topic", image_rgb_pub_topic, std::string("/vision_system/vsb/image_rgb_raw"));
    node_handle.param("/vision_system_bridge/publishers/image_depth/topic", image_depth_pub_topic, std::string("/vision_system/vsb/image_depth_raw"));
    node_handle.param("/vision_system_bridge/publishers/camera_info/topic", camera_info_pub_topic, std::string("/vision_system/vsb/camera_info"));

    node_handle.param("/vision_system_bridge/parameters/use_exact_time", use_exact_time, false);
}

void VisionSystemBridge::kinectCallback(const sensor_msgs::Image::ConstPtr &image_rgb, const sensor_msgs::Image::ConstPtr &image_depth, const sensor_msgs::CameraInfo::ConstPtr &camera_info)
{
    ROS_INFO("INPUT ID: rgb = %d, depth = %d,  info = %d", image_rgb->header.seq, image_depth->header.seq, camera_info->header.seq);

    publish(image_rgb, image_depth, camera_info);

    seq++;
}

void VisionSystemBridge::publish(const sensor_msgs::Image::ConstPtr &image_rgb_ptr, const sensor_msgs::Image::ConstPtr &image_depth_ptr, const sensor_msgs::CameraInfo::ConstPtr &camera_info_ptr)
{
    sensor_msgs::Image image_rgb = *(image_rgb_ptr);
    image_rgb.header.seq = seq;
    sensor_msgs::Image image_depth = *(image_depth_ptr);
    image_depth.header.seq = seq;
    sensor_msgs::CameraInfo camera_info = *(camera_info_ptr);
    camera_info.header.seq = seq;

    image_rgb_pub.publish(image_rgb);
    image_depth_pub.publish(image_depth);
    camera_info_pub.publish(camera_info);
}