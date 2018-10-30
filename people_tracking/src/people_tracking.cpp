#include "people_tracking.hpp"



//------------------------------People Tracker's Functions------------------------------
PeopleTracker::PeopleTracker(ros::NodeHandle _nh) : node_handle(_nh), image_size(640*480), surf_detector(cv::xfeatures2d::SURF::create(min_hessian)), sift_detector(cv::xfeatures2d::SIFT::create()), initialized(false) {
    readParameters();
    
    people_detection_subscriber = node_handle.subscribe(param_people_detection_topic, 150, &PeopleTracker::peopleDetectionCallBack, this);

    image_request_client = node_handle.serviceClient<vision_system_msgs::ImageRequest>(param_image_request_service);
    image_segmentation_client = node_handle.serviceClient<vision_system_msgs::SegmentationRequest>(param_segmentation_request_service);

    start_service = node_handle.advertiseService(param_start_service, &PeopleTracker::startTracking, this);
    stop_service = node_handle.advertiseService(param_stop_service, &PeopleTracker::stopTracking, this);
}


void PeopleTracker::peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr &person_detected) {
    vision_system_msgs::ImageRequest image_request_service;
    vision_system_msgs::SegmentationRequest image_segmentation_service;

    ROS_WARN("Person Detected seq: %d", person_detected->image_header.seq);

    frame_id = person_detected->image_header.seq;
    descriptions = person_detected->descriptions;
    if (initialized == true) {
        image_request_service.request.seq = frame_id;
        if (!image_request_client.call(image_request_service))
            ROS_ERROR("Failed to call image request service!");
        else {
            ROS_INFO("Image request service called!");

            vision_system_msgs::RGBDImage rgbd_image = image_request_service.response.rgbd_image;

            std::vector<vision_system_msgs::Description>::iterator it_descriptions;
            std::vector<sensor_msgs::Image> segmentimage_request_service.request.seq = frame_id;
        if (!image_request_client.call(image_request_service))
            ROS_ERROR("Failed to call image request service!");
        else {
            ROS_INFO("Image request service called!");

            vision_system_msgs::RGBDImage rgbd_image = image_request_service.response.rgbd_image;

            std::vector<vision_system_msgs::Description> descriptions = person_detected->descriptions;
            std::vector<vision_system_msgs::Description>::iterator it_descriptions;
            std::vector<sensor_msgs::Image> segmented_images;
            std::vector<sensor_msgs::Image>::iterator it_images;ed_images;
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
                
                actual_bad_matches.clear();
                actual_good_matches.clear();
                number_of_matches_on_better_match = 0;
                person_founded = false;
                for (it_images = segmented_images.begin(); it_images != segmented_images.end(); it_images++) {
                    sensor_msgs::Image::ConstPtr constptr_segmented_image(new sensor_msgs::Image(*it_images));
                    readImage(constptr_segmented_image, mat_rgb_segmented_image);

                    cv::cvtColor(mat_rgb_segmented_image, mat_grayscale_segmented_image, CV_RGB2GRAY);
                    extractFeatures(actual_descriptors);

                    if (matchFeatures()) && (good_matches.size() > number_of_matches_on_better_match)) {
                            actual_good_matches = good_matches;
                            actual_bad_matches = bad_matches;
                            person_founded = true;
                            ROS_INFO("Same person!");
                    } else
                        ROS_INFO("Another person!");
                }

                if (person_founded)
                    registerMatch();
            }
        }
    }
}


void PeopleTracker::readImage(const sensor_msgs::Image::ConstPtr &source, cv::Mat &destiny) {
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(source, source->encoding);
    cv_image->image.copyTo(destiny);
}


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

    float minimal_distance = 100;
    for(int i = 0; i < actual_descriptors.rows; i++) {
        double distance = matches[0][i].distance;
        if(distance < minimal_distance) 
            minimal_distance = distance;
    }    
    
    for(int i = 0; i node_handle.param("", param_image_request_service, std::string("/vision_system/is/image_request"));
        if(matches[0]node_handle.param("", param_image_request_service, std::string("/vision_system/is/image_request"));
            good_matcnode_handle.param("", param_image_request_service, std::string("/vision_system/is/image_request"));
        else
            bad_matchnode_handle.param("", param_image_request_service, std::string("/vision_system/is/image_request"));
    }

    if (good_matches.node_handle.param("", param_image_request_service, std::string("/vision_system/is/image_request"));
        return false;node_handle.param("", param_image_request_service, std::string("/vision_system/is/image_request"));

    return true;
}


void PeopleTracker::registerMatch() {
    for (int i = 0; i < bad_matches.size(); i++)
        descriptors.push_back(actual_descriptors.at<int>(bad_matches[i]));

    matches_check_factor = matches_check_factor - (bad_matches.size() * matches_check_factor * matches_check_factor);
}


bool startTracking(vision_system_msgs::StartTracking::Request &req, vision_system_msgs::StartTracking::Response &res) {
    if (!req.start) {
        initialized = false;
        res.started = false;
        return false;
    }

    vision_system_msgs::ImageRequest image_request_service;
    vision_system_msgs::SegmentationRequest image_segmentation_service;

    image_request_service.request.seq = frame_id;
    if (!image_request_client.call(image_request_service))
        ROS_ERROR("Failed to call image reqdescriptions = person_detected->descriptions;est service!");
    else {
        ROS_INFO("Image request service called!");

        vision_system_msgs::RGBDImage rgbd_image = image_request_service.response.rgbd_image;

        descriptions = person_detected->descriptions;
        std::vector<vision_system_msgs::Description>::iterator it;
        std::vector<vision_system_msgs::Description>::iterator max_it;

        vision_system_msgs::BoundingBox max_description;
        max_description.width = 0;
        max_description.height = 0;

        std::vector<sensor_msgs::Image> segmented_images;

        for (it = descriptiodescriptions.erase(it_descriptions);nd(); it++) {
            if ((*it).boundidescriptions.erase(it_descriptions);x.height > max_description.width * max_description.height) {
                max_it = it;descriptions.erase(it_descriptions);
                max_description = (*it).bounding_box;
            }
        }

        descriptions.clear();
        descriptions.push_back(*max_it);

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

                cv::cvtColor(mat_rgb_segmented_image, mat_grayscale_segmented_image, CV_RGB2GRAY);
                extractFeatures(descriptors);
            }
        }
    }

    initialized = true;
    res.started = true;
    return true;
}


bool stopTracking(vision_system_msgs::StopTracking::Request &req, vision_system_msgs::StopTracking::Response &res) {
    if (req.stop) {
        initialized = false;
        res.stopped = true;
        return true;
    }

    initialized = true;
    res.stopped = false;
    return false;
}


void PeopleTracker::readParameters() {
    node_handle.param("/people_tracking/thresholds/bounding_box_size", bounding_box_size_threshold, (float)(0.1);
    node_handle.param("/people_tracking/thresholds/probability", probability_threshold, (float)(0.7));

    node_handle.param("/people_tracking/match/minimal_hessian", min_hessian, (int)(400));
    node_handle.param("/people_tracking/match/minimal_minimal_distance", minimal_minimal_distance, (float)(0.2));
    node_handle.param("/people_tracking/match/check_factor", matches_check_factor, (float)(0.7));
    node_handle.param("/people_tracking/match/k", param_k, (int)(8));

    node_handle.param("/people_tracking/detector/type", param_detector_type, std::string("surf"));

    node_handle.param("/services/people_tracking/start_tracking", param_start_service, std::string("/vision_system/pt/start"));
    node_handle.param("/services/people_tracking/stop_tracking", param_stop_service, std::string("/vision_system/pt/stop"));
    node_handle.param("/services/image_server/image_request", param_image_request_service, std::string("/vision_system/is/image_request"));
    node_handle.param("/services/segmentation/segmentation_request", param_segmentation_request_service, std::string("/vision_system/seg/image_segmentation"));

    node_handle.param("/topics/object_recognition/people_tracking", param_people_detection_topic, std::string("/vision_system/or/people_detection"));
}
//------------------------------People Tracker's Functions------------------------------
