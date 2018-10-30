#include "people_tracking.hpp"



//------------------------------People Tracker's Functions------------------------------
PeopleTracker::PeopleTracker(ros::NodeHandle _nh) : node_handle(_nh), image_size(640*480), bounding_box_size_threshold(0.1), min_hessian(400)//, surf_detector(cv::xfeatures2d::SURF::create(min_hessian)), sift_detector(cv::xfeatures2d::SIFT::create()), param_detector_type("surf"), minimal_minimal_distance(0.02), matches_check_factor(0.7), initialized(false), param_k(8), probability_threshold(0.7) {
{
    people_detection_subscriber = node_handle.subscribe("/vision_system/or/people_detection", 150, &PeopleTracker::peopleDetectionCallBack, this);
    image_request_client = node_handle.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/is/image_request");
    image_segmentation_client = node_handle.serviceClient<vision_system_msgs::SegmentationRequest>("/vision_system/seg/image_segmentation");
}


void PeopleTracker::peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr &person_detected) {
    vision_system_msgs::ImageRequest image_request_service;
    vision_system_msgs::SegmentationRequest image_segmentation_service;

    ROS_WARN("Person Detected seq: %d", person_detected->image_header.seq);

    int frame_id = person_detected->image_header.seq;

    image_request_service.request.seq = frame_id;
    if (!image_request_client.call(image_request_service))
        ROS_ERROR("Failed to call image request service!");
    else {
        ROS_INFO("Image request service called!");

        vision_system_msgs::RGBDImage rgbd_image = image_request_service.response.rgbd_image;

        std::vector<vision_system_msgs::Description> descriptions = person_detected->descriptions;
        std::vector<vision_system_msgs::Description>::iterator it_descriptions;
        std::vector<sensor_msgs::Image> segmented_images;
        std::vector<sensor_msgs::Image>::iterator it_images;

        for (it_descriptions = descriptions.begin(); it_descriptions != descriptions.end(); it_descriptions++) {
            if (((*it_descriptions).bounding_box.width * (*it_descriptions).bounding_box.height < bounding_box_size_threshold * image_size) || ((*it_descriptions).probability < probability_threshold))
                descriptions.erase(it_descriptions);
        }

        image_segmentation_service.request.model_id = "median_full";
        image_segmentation_service.request.descriptions = descriptions;
        image_segmentation_service.request.initial_rgbd_image = rgbd_image;
        if (!image_segmentation_client.call(image_segmentation_service))
            ROS_ERROR("Failed to call image segmentation service!");
        else {
            ROS_INFO("Image segmentation service called!");
            segmented_images = image_segmentation_service.response.segmented_rgb_images;
            
            for (it_images = segmented_images.begin(); it_images != segmented_images.end(); it_images++) {
                sensor_msgs::Image::ConstPtr constptr_segmented_image(new sensor_msgs::Image(*it_images));
                readImage(constptr_segmented_image, mat_rgb_segmented_image);

                cv::imshow("Image", mat_rgb_segmented_image);
                cv::waitKey(1);
            }
            /*cv::cvtColor(mat_rgb_segmented_image, mat_grayscale_segmented_image, CV_RGB2GRAY);

            extractFeatures(actual_descriptors);
            if (matchFeatures())
                ROS_INFO("Person founded!");
            else
                ROS_INFO("Person not founded!");*/
        }
    }
}


void PeopleTracker::readImage(const sensor_msgs::Image::ConstPtr &source, cv::Mat &destiny) {
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(source, source->encoding);
    cv_image->image.copyTo(destiny);
}

/*
void PeopleTracker::extractFeatures(cv::Mat &descriptors_destiny) {
    keypoints.clear();
    descriptors_destiny = cv::Mat();

    if (param_detector_type == "surf")
        surf_detector->detectAndCompute(mat_grayscale_segmented_image, cv::Mat(), keypoints, descriptors_destiny);
    else if (param_detector_type == "sift")
        sift_detector->detectAndCompute(mat_grayscale_segmented_image, cv::Mat(), keypoints, descriptors_destiny);
}


bool PeopleTracker::matchFeatures() {
    matches.clear();
    good_matches.clear();
    bad_matches.clear();

    matcher.knnMatch(actual_descriptors, descriptors, matches, param_k);
    ROS_INFO("Fez o match!");

    float minimal_distance = 100;
    for(int i = 0; i < actual_descriptors.rows; i++) {
        double distance = matches[0][i].distance;
        if(distance < minimal_distance) 
            minimal_distance = distance;
    }    
    
    for(int i = 0; i < actual_descriptors.rows; i++) {
        if(matches[0][i].distance <= std::max(2 * minimal_distance, minimal_minimal_distance))
            good_matches.push_back(matches[0][i]);
        else
            bad_matches.push_back(i);
    }

    if (good_matches.size() < descriptors.rows * matches_check_factor)
        return false;

    registerMatch();
    return true;
}


void PeopleTracker::registerMatch() {
    for (int i = 0; i < bad_matches.size(); i++)
        descriptors.push_back(actual_descriptors.at<int>(bad_matches[i]));

    matches_check_factor = matches_check_factor - (bad_matches.size() * matches_check_factor * matches_check_factor);
}*/
//------------------------------People Tracker's Functions------------------------------
