#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/RGBDImage.h"

class ImgServer {
    private:
        std::vector<sensor_msgs::ImageConstPtr> rgb_buffer;
        std::vector<sensor_msgs::ImageConstPtr> depth_buffer;

        int min_seq;
        int max_seq;
        
        int last_rgb;
        int last_depth;

        bool unsynchronized;

        int buffer_size;

        ros::NodeHandle node_handle;

        ros::Subscriber img_rgb_sub;
        ros::Subscriber img_d_sub;
        ros::ServiceServer service;

    public:
        //Constructors
        ImgServer(ros::NodeHandle &nh);
        ImgServer(ros::NodeHandle &nh, int size);
        
        //Server
        bool accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res);

        //Callbacks
        void camCallBackRGB(const sensor_msgs::Image::ConstPtr& img);
        void camCallBackDepth(const sensor_msgs::Image::ConstPtr& img);
};