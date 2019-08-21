#include <ros/ros.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

class ButiaVisionBridge {
    public:
        ButiaVisionBridge(ros::NodeHandle &nh);

        void kinectCallback(const sensor_msgs::Image::ConstPtr &image_rgb, const sensor_msgs::Image::ConstPtr &image_depth, 
                            const sensor_msgs::CameraInfo::ConstPtr &camera_info, const sensor_msgs::PointCloud2::ConstPtr &points);
        
        void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr &camera_info, sensor_msgs::CameraInfo &info);
        void readImage(const sensor_msgs::Image::ConstPtr& msg_image, cv::Mat &image);

        void imageResize(cv::Mat &image);

        void publish(const sensor_msgs::Image::ConstPtr &image_rgb_ptr, const sensor_msgs::Image::ConstPtr &image_depth_ptr,
                     const sensor_msgs::CameraInfo::ConstPtr &camera_info_ptr, const sensor_msgs::PointCloud2::ConstPtr &points);

    private:
        ros::NodeHandle node_handle;

        bool use_exact_time;
        int sub_queue_size;
        int pub_queue_size;

        int image_width;
        int image_height;

        std::string image_rgb_sub_topic;
        std::string image_depth_sub_topic;
        std::string camera_info_sub_topic;
        std::string points_sub_topic;

        std::string image_rgb_pub_topic;
        std::string image_depth_pub_topic;
        std::string camera_info_pub_topic;
        std::string points_pub_topic;

        typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::PointCloud2> ExactSyncPolicy;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::PointCloud2> ApproximateSyncPolicy;

        image_transport::ImageTransport it;
        image_transport::SubscriberFilter *image_rgb_sub;
        image_transport::SubscriberFilter *image_depth_sub;
        message_filters::Subscriber<sensor_msgs::CameraInfo> *camera_info_sub;
        message_filters::Subscriber<sensor_msgs::PointCloud2> *points_sub;

        message_filters::Synchronizer<ExactSyncPolicy> *exact_sync;
        message_filters::Synchronizer<ApproximateSyncPolicy> *approximate_sync;

        image_transport::Publisher image_rgb_pub;
        image_transport::Publisher image_depth_pub;
        ros::Publisher camera_info_pub;
        ros::Publisher points_pub;

        long seq;

        void readParameters();
};