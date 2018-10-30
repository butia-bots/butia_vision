#include <ros/ros.h>
#include <opencv2/opencv.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>
#include <utility>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/BoundingBox.h"
#include "vision_system_msgs/RGBDImage.h"
#include "vision_system_msgs/Recognitions.h"
#include "vision_system_msgs/Description.h"
#include "vision_system_msgs/SegmentationRequest.h"



class PeopleTracker {
    private:
        ros::NodeHandle node_handle; //Internal NodeHandle
        ros::Subscriber people_detection_subscriber; //Subscriber that reads the people detection topic
        ros::ServiceClient image_request_client; //Service that gets the RGBD image
        ros::ServiceClient image_segmentation_client; //Service that segments the person

        int image_size;
        float bounding_box_size_threshold;
        float probability_threshold;

        cv::Mat mat_rgb_segmented_image;
        cv::Mat mat_grayscale_segmented_image;

        int min_hessian;
        //cv::Ptr<cv::xfeatures2d::SURF> surf_detector;
        //cv::Ptr<cv::xfeatures2d::SIFT> sift_detector;
        cv::FlannBasedMatcher matcher;

        cv::Mat descriptors;
        cv::Mat actual_descriptors;

        std::vector<cv::KeyPoint> keypoints;

        std::vector<std::vector<cv::DMatch>> matches;
        std::vector<std::vector<cv::DMatch>> actual_matches;

        std::vector<cv::DMatch> good_matches;
        std::vector<cv::DMatch> actual_good_matches;

        std::vector<int> bad_matches;
        std::vector<int> actual_bad_matches;

        std::string param_detector_type;
        int param_k;

        float minimal_minimal_distance;
        float matches_check_factor;

        bool initialized;
        bool person_founded;
        int number_of_matches_on_better_match;


    public:
        PeopleTracker(ros::NodeHandle _nh); //Constructor

        void peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr &person_detected); //CallBack

        void readImage(const sensor_msgs::Image::ConstPtr &source, cv::Mat &destiny);
        void extractFeatures(cv::Mat &descriptors_destiny);
        bool matchFeatures();
        void registerMatch();
};



