#include "image_server/image_server.h"

ImageServer::ImageServer(ros::NodeHandle &nh) : node_handle(nh), it(nh), min_seq(std::numeric_limits<int>::infinity()), max_seq(-1) 
{
    readParameters();
    
    resizeBuffers();

    image_rgb_sub = new image_transport::SubscriberFilter(it, image_rgb_sub_topic, sub_queue_size);
    image_depth_sub = new image_transport::SubscriberFilter(it, image_depth_sub_topic, sub_queue_size);
    camera_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(node_handle, camera_info_sub_topic, sub_queue_size);

    if(use_exact_time) {
        exact_sync = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(sub_queue_size), *image_rgb_sub, *image_depth_sub, *camera_info_sub);
        exact_sync->registerCallback(boost::bind(&ImageServer::imageCallback, this, _1, _2, _3));
    }
    else {
        approximate_sync = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(sub_queue_size),  *image_rgb_sub, *image_depth_sub, *camera_info_sub);
        approximate_sync->registerCallback(boost::bind(&ImageServer::imageCallback, this, _1, _2, _3));
    }

    image_request_server = node_handle.advertiseService(image_request_server_service, &ImageServer::imageRequestServer, this);
}

void ImageServer::resizeBuffers()
{
    image_rgb_buffer.resize(buffer_size);
    image_depth_buffer.resize(buffer_size);
    camera_info_buffer.resize(buffer_size);
}

void ImageServer::readParameters()
{
    node_handle.param("/image_server/subscribers/queue_size", sub_queue_size, 5);
    node_handle.param("/image_server/subscribers/image_rgb/topic", image_rgb_sub_topic, std::string("/vision_system/vsb/image_rgb_raw"));
    node_handle.param("/image_server/subscribers/image_depth/topic", image_depth_sub_topic, std::string("/vision_system/vsb/image_depth_raw"));
    node_handle.param("/image_server/subscribers/camera_info/topic", camera_info_sub_topic, std::string("/vision_system/vsb/camera_info"));

    node_handle.param("/image_server/server/image_request/service", image_request_server_service, std::string("/vision_system/is/image_request"));

    node_handle.param("/image_server/parameters/buffer_size", buffer_size, 150);
    node_handle.param("/image_server/parameters/use_exact_time", use_exact_time, false);
}

void ImageServer::imageCallback(const sensor_msgs::Image::ConstPtr &image_rgb, const sensor_msgs::Image::ConstPtr &image_depth, const sensor_msgs::CameraInfo::ConstPtr &camera_info)
{
    int seq = image_rgb->header.seq;
    ROS_INFO("INPUT ID: rgb = %d, depth = %d,  info = %d", image_rgb->header.seq, image_depth->header.seq, camera_info->header.seq);

    image_rgb_buffer[seq%buffer_size] = image_rgb;
    image_depth_buffer[seq%buffer_size] = image_depth;
    camera_info_buffer[seq%buffer_size] = camera_info;

    if(seq - buffer_size >= min_seq || seq < min_seq) min_seq = seq;
    max_seq = seq;
}

bool ImageServer::imageRequestServer(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res)
{
    int req_seq = req.seq;
    ROS_INFO("REQUEST ID: %d", req_seq);
    
    if(req_seq > max_seq || req_seq < min_seq) {
        return false;
    }

    res.rgbd_image.rgb = *(image_rgb_buffer[req_seq%buffer_size]);
    res.rgbd_image.rgb.header.seq = req_seq;
    
    res.rgbd_image.depth = *(image_depth_buffer[req_seq%buffer_size]);
    res.rgbd_image.depth.header.seq = req_seq;

    res.camera_info = *(camera_info_buffer[req_seq%buffer_size]);
    res.camera_info.header.seq = req_seq;
    
    return true;
}