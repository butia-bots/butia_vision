#include "ros/ros.h"
#include "ros/package.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <filesystem> 

#include "boost/filesystem.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>

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

#include "geometry_msgs/PoseWithCovariance.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h" //test

#include <cv_bridge/cv_bridge.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudNormal;

class Image2Kinect{
    public:
        Image2Kinect(ros::NodeHandle _nh);

        bool points2RGBPoseWithCovariance(PointCloud &points, butia_vision_msgs::BoundingBox &bb, geometry_msgs::PoseWithCovariance &pose, std_msgs::ColorRGBA &color, cv::Mat &mask);

        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image);
        void readPoints(const sensor_msgs::PointCloud2::ConstPtr& msg_points, PointCloud &points);

        void recognitions2Recognitions3d(butia_vision_msgs::Recognitions &recognitions, butia_vision_msgs::Recognitions3D &recognitions3d);

        void publishTF(butia_vision_msgs::Recognitions3D &recognitions3d);

        void objectRecognitionCallback(butia_vision_msgs::Recognitions recognitions);
        void faceRecognitionCallback(butia_vision_msgs::Recognitions recognitions);
        void peopleDetectionCallback(butia_vision_msgs::Recognitions recognitions);
        void peopleTrackingCallback(butia_vision_msgs::Recognitions recognitions);

        //void loadObjectClouds();
        //void refinePose(const PointCloud::Ptr &points, geometry_msgs::Pose &pose, std::string label);

    private:

        ros::NodeHandle node_handle;

        ros::Subscriber object_recognition_sub;
        ros::Subscriber face_recognition_sub;
        ros::Subscriber people_detection_sub;
        ros::Subscriber people_tracking_sub;

        ros::Publisher object_recognition_pub;
        ros::Publisher face_recognition_pub;
        ros::Publisher people_detection_pub;
        ros::Publisher people_tracking_pub;

        ros::ServiceClient image_request_client;
        //ros::ServiceClient segmentation_request_client;

        int sub_queue_size;
        int pub_queue_size;

        //std::map<std::string, PointCloudNormal> object_clouds;

        std::string object_recognition_sub_topic;
        std::string face_recognition_sub_topic;
        std::string people_detection_sub_topic;
        std::string people_tracking_sub_topic;

        std::string object_recognition_pub_topic;
        std::string face_recognition_pub_topic;
        std::string people_detection_pub_topic;
        std::string people_tracking_pub_topic;

        std::string image_request_client_service;
        //std::string segmentation_request_client_service;

        //float segmentation_threshold;
        int max_depth;

        bool publish_tf;

        int width;
        int height;

        int kernel_size;

        //std::string segmentation_model_id;

        void readParameters();
};
