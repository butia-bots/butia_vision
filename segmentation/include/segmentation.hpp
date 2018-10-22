#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/ImageSegmentation.h"
#include "vision_system_msgs/BoundingBox.h"


enum Verifier {
    NOT_VERIFIED = 0, TRUTH = 1, LIE = 2
};


class ImageSegmenter {
    private:
        ros::NodeHandle node_handle;
        ros::ServiceServer service;

        sensor_msgs::Image::ConstPtr constptr_segmented_image;

        cv::Mat_<cv::Vec3b> mat_initial_rgb_image;
        cv::Mat_<uint16_t> mat_initial_depth_image;

        int bounding_box_size;
        int histogram_size;
        int upper_histogram_limit;
        int lower_histogram_limit;
        float *range;
        const float **histogram_range;

        float bounding_box_threshold;
        float histogram_decrease_factor;

        std::vector<int> histogram;
        std::vector<std::pair<int, int>> histogram_class_limits;
        int max_histogram_value;
        int position_of_max_value;

        cv::Mat_<uint8_t> mask;
        cv::Mat_<uint8_t> state;

        int *d;

    
    public:
        ImageSegmenter(ros::NodeHandle _nh); //Constructor
        
        bool segment(vision_system_msgs::ImageSegmentation::Request &req, vision_system_msgs::ImageSegmentation::Response &res); //Service function

        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image);//Image Reader
        void cropImage(cv::Mat &image, vision_system_msgs::BoundingBox bounding_box); //Image Cropper
        void calculateHistogram();
        void getMaxHistogramValue();
        void createMask();
        bool verifyState(int r, int c);
};