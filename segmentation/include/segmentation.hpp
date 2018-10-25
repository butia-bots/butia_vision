#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <string>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/SegmentationRequest.h"
#include "vision_system_msgs/BoundingBox.h"
#include "vision_system_msgs/Description.h"


enum Verifier {
    NOT_VERIFIED = 0, TRUTH = 1, LIE = 2, VERIFYING = 3
};


class ImageSegmenter {
    private:
        ros::NodeHandle node_handle;
        ros::ServiceServer service;

        std::vector<vision_system_msgs::Description> descriptions;
        std::vector<vision_system_msgs::Description>::iterator it;

        cv::Mat_<cv::Vec3b> mat_initial_rgb_image;
        cv::Mat_<uint16_t> mat_initial_depth_image;
        cv::Mat_<cv::Vec3b> cropped_initial_rgb_image;
        cv::Mat_<uint16_t> cropped_initial_depth_image;

        cv::Mat_<cv::Vec3b> mat_segmented_image;

        cv_bridge::CvImage ros_segmented_rgb_image;
        sensor_msgs::Image ros_segmented_msg_image;

        int histogram_size;
        int upper_histogram_limit;
        int lower_histogram_limit;
        int left_class_limit;
        int right_class_limit;
        float bounding_box_threshold;
        float histogram_decrease_factor;

        std::vector<int> histogram;
        std::vector<std::pair<int, int>> histogram_class_limits;

        int max_histogram_value;
        int position_of_max_value;

        cv::Mat_<uint8_t> mask;
        int *dif;

        std::string param_segmentation_service;

    
    public:
        ImageSegmenter(ros::NodeHandle _nh); //Constructor
        
        bool segment(vision_system_msgs::SegmentationRequest::Request &req, vision_system_msgs::SegmentationRequest::Response &res); //Service function

        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image); //Image Reader
        void cropImage(cv::Mat &image, vision_system_msgs::BoundingBox bounding_box, cv::Mat &destiny); //Image Cropper
        void calculateHistogram();
        void getMaxHistogramValue();
        void createMask();
        bool verifyState(int r, int c);

        void readParameters();
};