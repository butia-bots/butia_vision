#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include "opencv2/xfeatures2d.hpp"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/BoundingBox.h"
#include "vision_system_msgs/RGBDImage.h"
#include "vision_system_msgs/Recognitions.h"
#include "vision_system_msgs/Description.h"
#include "vision_system_msgs/SegmentationRequest.h"
#include "vision_system_msgs/StartTracking.h"
#include "vision_system_msgs/StopTracking.h"



class PeopleTracker {
    private:
        ros::NodeHandle node_handle; //Internal NodeHandle

        ros::Subscriber people_detection_subscriber; //Subscriber that reads the people detection topic

        ros::ServiceClient image_request_client; //Service that gets the RGBD image
        ros::ServiceClient image_segmentation_client; //Service that segments the person

        ros::ServiceServer start_service; //Service that starts the tracking
        ros::ServiceServer stop_service; //Service that stops the tracking

        int image_size; //Variable that stores the size of the camera image

        int frame_id; //Stores the frame id to request the image from server
        std::vector<vision_system_msgs::Description> descriptions; //Stores the descriptions of the people detected

        //Stores the threshold to know if is confiable
        float bounding_box_size_threshold;
        float probability_threshold;

        //Stores the cv mat segmented images
        cv::Mat mat_rgb_segmented_image;
        cv::Mat mat_grayscale_segmented_image;
        cv::Mat actual_better_segmented_image;

        int min_hessian; //Stores the hessian threshold to create the SURF detector
        cv::FlannBasedMatcher matcher; //Matcher

        //Detectors
        cv::Ptr<cv::xfeatures2d::SURF> surf_detector;
        cv::Ptr<cv::xfeatures2d::SIFT> sift_detector;

        //Descriptors
        std::vector<cv::Mat_<float>> descriptors;
        cv::Mat_<float> actual_descriptors;
        cv::Mat_<float> better_descriptors;

        std::vector<cv::KeyPoint> keypoints; //Keypoints

        //Matches
        std::vector<cv::DMatch> matches;
        int good_matches;

        //Parameters
        std::string param_detector_type;
        std::string param_start_service;
        std::string param_stop_service;
        std::string param_people_detection_topic;
        std::string param_image_request_service;
        std::string param_segmentation_request_service;
        int param_k;
        float minimal_minimal_distance;
        float matches_check_factor;
        int queue_size;

        //Control variables
        bool initialized;
        bool person_founded;

        //Queue description variables
        int queue_actual_size;
        int actual_iterator;

        int number_of_matches_on_better_match; //Stores the number of matches of better match


    public:
        PeopleTracker(ros::NodeHandle _nh); //Constructor

        void peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr &person_detected); //CallBack

        //Services' Functions
        bool startTracking(vision_system_msgs::StartTracking::Request &req, vision_system_msgs::StartTracking::Response &res);
        bool stopTracking(vision_system_msgs::StopTracking::Request &req, vision_system_msgs::StopTracking::Response &res);
        
        void readImage(const sensor_msgs::Image::ConstPtr &source, cv::Mat &destiny); //Function that reads the image

        //Feature matching functions  
        void extractFeatures(cv::Mat_<float> &destiny);
        bool matchFeatures(cv::Mat_<float> &destiny);
        void registerMatch();

        void readParameters(); //Function that reads the parameters
};