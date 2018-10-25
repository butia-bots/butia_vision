#include "ros/ros.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/Recognitions.h"
#include "vision_system_msgs/Recognitions3D.h"
#include "vision_system_msgs/Description.h"
#include "vision_system_msgs/Description3D.h"
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

        void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr &camera_info);
        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image);
        
        void recognitions2Recognitions3d(vision_system_msgs::Recognitions &recognitions, vision_system_msgs::Recognitions3D &recognitions3d);

        void objectRecognitionCallback(vision_system_msgs::Recognitions recognitions);
        void faceRecognitionCallback(vision_system_msgs::Recognitions recognitions);
        void peopleTrackingCallback(vision_system_msgs::Recognitions recognitions);

    private:
        ros::NodeHandle node_handle;

        ros::Subscriber object_recognition_sub;
        ros::Subscriber face_recognition_sub;
        ros::Subscriber people_tracking_sub;

        ros::Publisher object_recognition_pub;
        ros::Publisher face_recognition_pub;
        ros::Publisher people_tracking_pub;

        ros::ServiceClient image_request_client;
        ros::ServiceClient segmentation_client;


        int sub_queue_size;
        int pub_queue_size;

        std::string object_recognition_sub_topic;
        std::string face_recognition_sub_topic;
        std::string people_tracking_sub_topic;

        std::string object_recognition_pub_topic;
        std::string face_recognition_pub_topic;
        std::string people_tracking_pub_topic;

        std::string image_request_client_service;
        std::string segmentation_client_service;

        cv::Mat camera_matrix_color;

        cv::Mat table_x;
        cv::Mat table_y;

        int width;
        int height;

        void createTabels();
        void readParameters();
};
