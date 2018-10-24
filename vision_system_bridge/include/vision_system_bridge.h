#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/ImageRequest.h"

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

class VisionSystemBridge {
    public:
        VisionSystemBridge(ros::NodeHandle &nh);
        
        /*
        bool accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res);

        void camCallBackRGB(const sensor_msgs::ImageConstPtr img);
        void camCallBackDepth(const sensor_msgs::ImageConstPtr img);*/

    private:
        ros::NodeHandle node_handle;


        std::vector<sensor_msgs::ImageConstPtr> image_rgb_buffer;
        std::vector<sensor_msgs::ImageConstPtr> image_depth_buffer;
        std::vector<sensor_msgs::CameraInfo> camera_info_buffer;


        bool use_exact_time;
        int buffer_size;

        std::string image_rgb_topic;
        int image_rgb_qs;

        std::string image_depth_topic;
        int image_depth_qs;

        std::string camera_info_topic;
        int camera_info_qs;

        std::string rgbdimage_request_service;


        typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ExactSyncPolicy;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> ApproximateSyncPolicy;

        image_transport::SubscriberFilter *image_rgb_sub;
        image_transport::SubscriberFilter *image_depth_sub;
        message_filters::Subscriber<sensor_msgs::CameraInfo> *camera_info_sub;

        message_filters::Synchronizer<ExactSyncPolicy> *exact_sync;
        message_filters::Synchronizer<ApproximateSyncPolicy> *approximate_sync;

         /*
        int min_seq;
        int max_seq;
        
        int last_rgb;
        int last_depth;

        ros::Subscriber img_rgb_sub;
        ros::Subscriber img_d_sub;
        ros::ServiceServer service;
        */

        void resizeBuffers();
        void readParameters();
};