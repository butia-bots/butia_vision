#include "ros/ros.h"

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "vision_system_msgs/Image2World.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/Recognitions.h"
#include "vision_system_msgs/Description.h"
#include "vision_system_msgs/RGBDImage.h"

#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud.h"
#include "sensor_msgs/CameraInfo.h"

#include <cv_bridge/cv_bridge.h>

class Image2World{
    public:
        Image2World(ros::NodeHandle _nh);

        void rgb2PointCloud(cv::Mat &color, cv::Mat &depth, sensor_msgs::PointCloud& point_cloud);

        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image);
        
        void createTabels();

        void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &camera_info);
        bool image2worldCallback(vision_system_msgs::Image2World::Request &request, vision_system_msgs::Image2World::Response &response);

    private:
        ros::NodeHandle node_handle;

        ros::Subscriber camera_info_subscriber;
        ros::ServiceServer image2world_server;
        ros::ServiceClient image_client;
        ros::ServiceClient segmentation_client;

        std::string camera_info_topic;
        int camera_info_qs;

        std::string image2world_server_service;
        std::string image_client_service;
        std::string segmentation_client_service;

        cv::Mat camera_matrix_color;

        cv::Mat table_x;
        cv::Mat table_y;

        int width;
        int height;

        void readParameters();
};