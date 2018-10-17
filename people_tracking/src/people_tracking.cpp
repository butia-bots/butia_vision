#include "people_tracking.hpp"



//------------------------------People Tracker's Functions------------------------------
PeopleTracker::PeopleTracker(ros::NodeHandle _nh) : node_handle(_nh) {
    people_detection_subscriber = node_handle.subscribe("/vision_system/or/people_detection", 150, &PeopleTracker::peopleDetectionCallBack, this);
    image_client = node_handle.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/is/image_request");
}


void PeopleTracker::peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr& person_detected) {
    vision_system_msgs::ImageRequest image_service;
    image_service.request.frame = person_detected->image_header.seq;

    if (!image_client.call(image_service))
        ROS_ERROR("Failed to call image_server service");
    else {
        cv::Mat rgb_image, depth_image;
        vision_system_msgs::RGBDImage rgbd_image = image_service.response.rgbd_image;

        sensor_msgs::Image::ConstPtr rgb_const_ptr(new sensor_msgs::Image(rgbd_image.rgb));
        sensor_msgs::Image::ConstPtr depth_const_ptr(new sensor_msgs::Image(rgbd_image.depth));

        readImage(rgb_const_ptr, rgb_image);
        readImage(depth_const_ptr, depth_image);
    }
}


void PeopleTracker::readImage(const sensor_msgs::Image::ConstPtr& msg_image, cv::Mat &image) {
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(msg_image, msg_image->encoding);
    cv_image->image.copyTo(image);
}
//------------------------------People Tracker's Functions------------------------------































/*cv::Mat PeopleTracker::getMask(const cv::Mat depth_image, int bounding_box_size) {
    cv::Mat histogram;
    int histogram_size = 64;
    float range[] = {0, 5001};
    const float *histogram_range = {range};
    bool accumulate = false;
    bool uniform = true;
    const int channels[] = {0};
    float bounding_box_threshold = 0.65;

    int number_of_pixels;
    do {
        cv::calcHist(&depth_image, 1, channels, cv::Mat(), histogram, 1, &histogram_size, &histogram_range, uniform, accumulate);
        number_of_pixels = getMax(histogram);
        if (number_of_pixels < bounding_box_size * bounding_box_threshold)
            histogram_size = histogram_size / 1.5;
    } while (number_of_pixels < bounding_box_size * bounding_box_threshold);
    int reference_value = histogram.at<int>((int)(histogram_range[1] / histogram_size), 0) * number_of_pixels; //Defining the reference depth value
    int mask_threshold = (histogram_range[1] / histogram_size) / 2; //Defining the threshold

    cv::Mat mask;
    for (int i = 0; i < depth_image.rows; i++) {
        for (int j = 0; j < depth_image.cols; j++) {
            if ((depth_image.at<int>(i, j) >= reference_value - mask_threshold) && (depth_image.at<int>(i, j) <= reference_value + mask_threshold))
                mask.at<int>(i, j) = 1;
            else
                mask.at<int>(i, j) = 0;
        }
    }
    return mask;
}*/