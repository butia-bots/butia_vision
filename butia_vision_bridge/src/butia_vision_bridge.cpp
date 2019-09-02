#include "butia_vision_bridge/butia_vision_bridge.h"

ButiaVisionBridge::ButiaVisionBridge(ros::NodeHandle &nh) : node_handle(nh), it(nh), seq(0)
{
    readParameters();
    
    image_rgb_sub = new image_transport::SubscriberFilter(it, image_rgb_sub_topic, sub_queue_size);
    image_depth_sub = new image_transport::SubscriberFilter(it, image_depth_sub_topic, sub_queue_size);
    camera_info_sub = new message_filters::Subscriber<sensor_msgs::CameraInfo>(node_handle, camera_info_sub_topic, sub_queue_size);
    points_sub = new message_filters::Subscriber<sensor_msgs::PointCloud2>(node_handle, points_sub_topic, sub_queue_size);

    if(use_exact_time) {
        exact_sync = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(sub_queue_size), *image_rgb_sub,
                                                                        *image_depth_sub, *camera_info_sub, *points_sub);
        exact_sync->registerCallback(boost::bind(&ButiaVisionBridge::kinectCallback, this, _1, _2, _3, _4));
    }
    else {
        approximate_sync = new message_filters::Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(sub_queue_size),  *image_rgb_sub,
                                                                                    *image_depth_sub, *camera_info_sub, *points_sub);
        approximate_sync->registerCallback(boost::bind(&ButiaVisionBridge::kinectCallback, this, _1, _2, _3, _4));
    }

    image_rgb_pub = it.advertise(image_rgb_pub_topic, pub_queue_size);
    image_depth_pub = it.advertise(image_depth_pub_topic, pub_queue_size);
    camera_info_pub = node_handle.advertise<sensor_msgs::CameraInfo>(camera_info_pub_topic, pub_queue_size);
    points_pub = node_handle.advertise<sensor_msgs::PointCloud2>(points_pub_topic, pub_queue_size);
}

void ButiaVisionBridge::readParameters()
{
    node_handle.param("/butia_vision_bridge/subscribers/queue_size", sub_queue_size, 5);
    node_handle.param("/butia_vision_bridge/subscribers/image_rgb/topic", image_rgb_sub_topic, std::string("/kinect2/qhd/image_color_rect"));
    node_handle.param("/butia_vision_bridge/subscribers/image_depth/topic", image_depth_sub_topic, std::string("/kinect2/qhd/image_depth_rect"));
    node_handle.param("/butia_vision_bridge/subscribers/camera_info/topic", camera_info_sub_topic, std::string("/kinect2/qhd/camera_info"));
    node_handle.param("/butia_vision_bridge/subscribers/points/topic", points_sub_topic, std::string("/kinect2/qhd/points"));

    node_handle.param("/butia_vision_bridge/publishers/queue_size", pub_queue_size, 5);   
    node_handle.param("/butia_vision_bridge/publishers/image_rgb/topic", image_rgb_pub_topic, std::string("/butia_vision/bvb/image_rgb_raw"));
    node_handle.param("/butia_vision_bridge/publishers/image_depth/topic", image_depth_pub_topic, std::string("/butia_vision/bvb/image_depth_raw"));
    node_handle.param("/butia_vision_bridge/publishers/camera_info/topic", camera_info_pub_topic, std::string("/butia_vision/bvb/camera_info"));
    node_handle.param("/butia_vision_bridge/publishers/points/topic", points_pub_topic, std::string("/butia_vision/bvb/points"));

    node_handle.param("/butia_vision_bridge/use_exact_time", use_exact_time, false);
    node_handle.param("/butia_vision_bridge/image_width", image_width, 640);
    node_handle.param("/butia_vision_bridge/image_height", image_height, 480);
}

void ButiaVisionBridge::readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr &camera_info, sensor_msgs::CameraInfo &info)
{
    info = *(camera_info);

    float scale_x = (image_width) / (float)(info.width);
    float scale_y = (image_height) / (float)(info.height);

    info.K[0] *= scale_x;
    info.K[2] *= scale_x;

    info.K[4] *= scale_y;
    info.K[5] *= scale_y;

    info.width = image_width;
    info.height = image_height;
}

void ButiaVisionBridge::readPointCloud(const sensor_msgs::PointCloud2::ConstPtr& msg_points, sensor_msgs::PointCloud2 &points)
{
    points = *(msg_points);
}


void ButiaVisionBridge::readImage(const sensor_msgs::Image::ConstPtr& msg_image, cv::Mat &image)
{
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(msg_image, msg_image->encoding);
    cv_image->image.copyTo(image);
}

void ButiaVisionBridge::imageResize(cv::Mat &image)
{
    cv::Mat cp_image;

    image.copyTo(cp_image);

    cv::resize(cp_image, image, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);
}

void ButiaVisionBridge::kinectCallback(const sensor_msgs::Image::ConstPtr &image_rgb_ptr, const sensor_msgs::Image::ConstPtr &image_depth_ptr,
                                       const sensor_msgs::CameraInfo::ConstPtr &camera_info_ptr, const sensor_msgs::PointCloud2::ConstPtr &points_ptr)
{
    ROS_INFO("INPUT ID: rgb = %d, depth = %d, info = %d, points = %d", image_rgb_ptr->header.seq, image_depth_ptr->header.seq, camera_info_ptr->header.seq, points_ptr->header.seq);

    cv::Mat rgb_image, depth_image;
    sensor_msgs::CameraInfo camera_info;
    sensor_msgs::PointCloud2 points;

    readImage(image_rgb_ptr, rgb_image);
    imageResize(rgb_image);
    readImage(image_depth_ptr, depth_image);
    imageResize(depth_image);

    readCameraInfo(camera_info_ptr, camera_info);

    readPointCloud(points_ptr, points);

    cv_bridge::CvImage rgb_cv_image(image_rgb_ptr->header, image_rgb_ptr->encoding, rgb_image);
    sensor_msgs::Image rgb_image_message;
    rgb_cv_image.toImageMsg(rgb_image_message);
    rgb_image_message.header.seq = seq;

    cv_bridge::CvImage depth_cv_image(image_depth_ptr->header, image_depth_ptr->encoding, depth_image);
    sensor_msgs::Image depth_image_message;
    depth_cv_image.toImageMsg(depth_image_message);
    depth_image_message.header.seq = seq;

    camera_info.header.seq = seq;

    points.header.seq = seq;
    
    publish(rgb_image_message, depth_image_message, camera_info, points);

    seq++;
}

void ButiaVisionBridge::publish(sensor_msgs::Image &rgb_image_message, sensor_msgs::Image &depth_image_message,
                                sensor_msgs::CameraInfo &camera_info_message, sensor_msgs::PointCloud2 &points_message)
{
    image_rgb_pub.publish(rgb_image_message);
    image_depth_pub.publish(depth_image_message);
    camera_info_pub.publish(camera_info_message);
    points_pub.publish(points_message);
}