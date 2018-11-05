#include <ros/ros.h>
#include <vector>

#include "sensor_msgs/Image.h"
#include "vision_system_msgs/ImageRequest.h"

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

class ImageServer {
    public:
        ImageServer(ros::NodeHandle &nh);

        void imageCallback(const sensor_msgs::Image::ConstPtr &image_rgb, const sensor_msgs::Image::ConstPtr &image_depth, const sensor_msgs::CameraInfo::ConstPtr &camera_info);

        bool imageRequestServer(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res);

    private:
        ros::NodeHandle node_handle;

        std::vector<sensor_msgs::Image::ConstPtr> image_rgb_buffer;
        std::vector<sensor_msgs::Image::ConstPtr> image_depth_buffer;
        std::vector<sensor_msgs::CameraInfo::ConstPtr> camera_info_buffer;
        
        bool use_exact_time;
        int buffer_size;
        int sub_queue_size;
        int pub_queue_size;

        std::string image_rgb_sub_topic;
        std::string image_depth_sub_topic;
        std::string camera_info_sub_topic;

        std::string image_request_server_service;

        typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ExactSyncPolicy;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ApproximateSyncPolicy;

        image_transport::ImageTransport it;
        image_transport::SubscriberFilter *image_rgb_sub;
        image_transport::SubscriberFilter *image_depth_sub;
        message_filters::Subscriber<sensor_msgs::CameraInfo> *camera_info_sub;

        message_filters::Synchronizer<ExactSyncPolicy> *exact_sync;
        message_filters::Synchronizer<ApproximateSyncPolicy> *approximate_sync;

        ros::ServiceServer image_request_server;

        int min_seq, max_seq;

        void resizeBuffers();
        void readParameters();
};