#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>
#include <utility>
#include <string>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "butia_vision_msgs/SegmentationRequest.h"
#include "butia_vision_msgs/BoundingBox.h"
#include "butia_vision_msgs/Description.h"


enum Verifier {
    NOT_VERIFIED = 0, TRUTH = 1, LIE = 2, VERIFYING = 3
};


class ImageSegmenter {
    private:
        ros::NodeHandle node_handle;
        ros::ServiceServer service;

        ros::Publisher segmented_image_pub;

        cv::Mat_<cv::Vec3b> mat_initial_rgb_image, cropped_initial_rgb_image;
        cv::Mat_<uint16_t> mat_initial_depth_image, cropped_initial_depth_image;
        cv::Mat_<uint8_t> mask;

        int *dif;

        std::string model_id;
        
        int upper_limit;
        int lower_limit;
        int left_class_limit;
        int right_class_limit;

        int histogram_size;
        float bounding_box_threshold;
        float histogram_decrease_factor;
        std::vector<int> histogram;
        std::vector<std::pair<int, int>> histogram_class_limits;
        int max_histogram_value;
        int position_of_max_value;

        bool median_outlier_rejection;
        int median_full_threshold;

        int median_center_kernel_size;
        int median_center_threshold;

        int segmentable_depth;

        std::string param_segmentation_service;
        std::string image_segmentation_view_topic;
        int image_segmentation_view_qs;

    
    public:
        ImageSegmenter(ros::NodeHandle _nh); //Constructor
        
        bool segment(butia_vision_msgs::SegmentationRequest::Request &req, butia_vision_msgs::SegmentationRequest::Response &res); //Service function
	    void filterImage(cv::Mat &image);
        void readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image); //Image Reader
        void cropImage(cv::Mat &image, butia_vision_msgs::BoundingBox bounding_box, cv::Mat &destiny); //Image Cropper
        void calculateHistogram();
        void getMaxHistogramValue();
        
        void createMask(std::string _model_id);
        void createMaskHistogram();
        void createMaskMedianFull();
        void createMaskMedianCenter();

        bool verifyStateHistogram(int r, int c);
        bool verifyStateMedianFull(int r, int c);
        bool verifyStateMedianCenter(int r, int c);

        void readParameters();
};
