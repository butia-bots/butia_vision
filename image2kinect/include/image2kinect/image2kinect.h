#include "ros/ros.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <map>

#include "butia_vision_msgs/Recognitions.h"
#include "butia_vision_msgs/Recognitions3D.h"
#include "butia_vision_msgs/Description.h"
#include "butia_vision_msgs/Description3D.h"
#include "butia_vision_msgs/RGBDImage.h"
#include "butia_vision_msgs/ImageRequest.h"
#include "butia_vision_msgs/SegmentationRequest.h"

#include <tf/transform_broadcaster.h>

#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

#include "geometry_msgs/Point.h"

#include <cv_bridge/cv_bridge.h>

class Image2Kinect{
    public:
        Image2Kinect(ros::NodeHandle _nh);

        bool rgbd2RGBPoint(cv::Mat &image_color, cv::Mat &image_depth, geometry_msgs::Point &point, std_msgs::ColorRGBA &color, int x_offset, int y_offset);

        void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr &camera_info);
        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image);
        
        void recognitions2Recognitions3d(butia_vision_msgs::Recognitions &recognitions, butia_vision_msgs::Recognitions3D &recognitions3d);

        void publishTF(butia_vision_msgs::Recognitions3D &recognitions3d);

        void objectRecognitionCallback(butia_vision_msgs::Recognitions recognitions);
        void faceRecognitionCallback(butia_vision_msgs::Recognitions recognitions);
        void peopleTrackingCallback(butia_vision_msgs::Recognitions recognitions);

    private:
        ros::NodeHandle node_handle;

        ros::Subscriber object_recognition_sub;
        ros::Subscriber face_recognition_sub;
        ros::Subscriber people_tracking_sub;

        ros::Publisher object_recognition_pub;
        ros::Publisher face_recognition_pub;
        ros::Publisher people_tracking_pub;

        ros::ServiceClient image_request_client;
        ros::ServiceClient segmentation_request_client;

        int sub_queue_size;
        int pub_queue_size;

        std::string object_recognition_sub_topic;
        std::string face_recognition_sub_topic;
        std::string people_tracking_sub_topic;

        std::string object_recognition_pub_topic;
        std::string face_recognition_pub_topic;
        std::string people_tracking_pub_topic;

        std::string image_request_client_service;
        std::string segmentation_request_client_service;

        cv::Mat camera_matrix_color;

        cv::Mat table_x;
        cv::Mat table_y;

        int width;
        int height;

        std::string segmentation_model_id;

        void createTabels();
        void readParameters();
};
