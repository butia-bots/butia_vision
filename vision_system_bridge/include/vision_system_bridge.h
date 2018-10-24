#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/ImageRequest.h"

class VisionSystemBridge {
    public:
        VisionSystemBridge(ros::NodeHandle &nh);
        
        /*
        bool accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res);

        void camCallBackRGB(const sensor_msgs::ImageConstPtr img);
        void camCallBackDepth(const sensor_msgs::ImageConstPtr img);*/

    private:
        ros::NodeHandle node_handle;
        
        int buffer_size;

        std::vector<sensor_msgs::ImageConstPtr> rgb_buffer;
        std::vector<sensor_msgs::ImageConstPtr> depth_buffer;

        int min_seq;
        int max_seq;
        
        int last_rgb;
        int last_depth;

        std::string image_rgb_topic;
        int image_rgb_qs;

        std::string image_depth_topic;
        int image_depth_qs;

        std::string rgbdimage_request_service;

        ros::Subscriber img_rgb_sub;
        ros::Subscriber img_d_sub;
        ros::ServiceServer service;

        void readParameters();
};