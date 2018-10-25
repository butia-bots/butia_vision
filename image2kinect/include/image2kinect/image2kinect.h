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

#include "geometry_msgs/PoseWithCovariance.h"

#include <cv_bridge/cv_bridge.h>

class Image2Kinect{
    public:
        Image2Kinect(ros::NodeHandle _nh);

        void rgbd2PoseWithCovariance(cv::Mat &color, cv::Mat &depth, geometry_msgs::PoseWithCovariance &pose);

        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image);

        void createTabels();

        void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &camera_info);
        bool image2kinectCallback(vision_system_msgs::Image2Kinect::Request &request, vision_system_msgs::Image2Kinect::Response &response);

    private:
        ros::NodeHandle node_handle;

        ros::Subscriber camera_info_subscriber;
        ros::ServiceServer image2kinect_server;
        ros::ServiceClient image_client;
        ros::ServiceClient segmentation_client;

        std::string camera_info_topic;
        int camera_info_qs;

        std::string image2kinect_server_service;
        std::string image_client_service;
        std::string segmentation_client_service;

        cv::Mat camera_matrix_color;

        cv::Mat table_x;
        cv::Mat table_y;

        int width;
        int height;

        void readParameters();
};
