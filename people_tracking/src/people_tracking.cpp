#include "people_tracking.hpp"

//-------------------------------------People Tracker's Functions-------------------------------------
PeopleTracker::PeopleTracker(ros::NodeHandle _nh) : node_handle(_nh), image_size(640*480), initialized(false), actual_iterator(0), queue_actual_size(0) {
    readParameters();
    
    people_detection_subscriber = node_handle.subscribe(param_people_detection_topic, 150, &PeopleTracker::peopleDetectionCallBack, this);

    people_tracking_publisher = node_handle.advertise<butia_vision_msgs::Recognitions>(param_people_tracking_topic, 1000);

    image_request_client = node_handle.serviceClient<butia_vision_msgs::ImageRequest>(param_image_request_service);
    image_segmentation_client = node_handle.serviceClient<butia_vision_msgs::SegmentationRequest>(param_segmentation_request_service);

    start_service = node_handle.advertiseService(param_start_service, &PeopleTracker::startTracking, this);
    stop_service = node_handle.advertiseService(param_stop_service, &PeopleTracker::stopTracking, this);

    surf_detector = cv::xfeatures2d::SURF::create(min_hessian);
    sift_detector = cv::xfeatures2d::SIFT::create();

    descriptors.resize(queue_size);
}


bool PeopleTracker::startTracking(butia_vision_msgs::StartTracking::Request &req, butia_vision_msgs::StartTracking::Response &res) {
    if (!req.start) {
        initialized = false;
        res.started = false;
        return false;
    }

    butia_vision_msgs::ImageRequest image_request_service;
    butia_vision_msgs::SegmentationRequest image_segmentation_service;

    image_request_service.request.seq = frame_id;
    if (!image_request_client.call(image_request_service)) {
        ROS_ERROR("Failed to call image request service!");
        ROS_ERROR("Failed to start tracking!");
        initialized = false;
        res.started = false;
        return false;
    }
    
    ROS_INFO("Image request service called!");

    butia_vision_msgs::RGBDImage rgbd_image = image_request_service.response.rgbd_image;
    std::vector<butia_vision_msgs::Description>::iterator it;
    std::vector<butia_vision_msgs::Description>::iterator max_it;

    butia_vision_msgs::BoundingBox max_description;
    max_description.width = 0;
    max_description.height = 0;

    std::vector<sensor_msgs::Image> segmented_images;

    for (it = descriptions.begin(); it != descriptions.end(); it++) {
        if ((*it).bounding_box.width * (*it).bounding_box.height > max_description.width * max_description.height) {
            max_it = it;
            max_description = (*it).bounding_box;
        }
    }

    descriptions.clear();
    descriptions.push_back(*max_it);

    image_segmentation_service.request.model_id = "histogram";
    image_segmentation_service.request.descriptions = descriptions;
    image_segmentation_service.request.initial_rgbd_image = rgbd_image;
    if (!image_segmentation_client.call(image_segmentation_service)) {
        ROS_ERROR("Failed to call image segmentation service!");
        ROS_ERROR("Failed to start tracking!");
        initialized = false;
        res.started = false;
        return false;
    }

    ROS_INFO("Image segmentation service called!");
    segmented_images = image_segmentation_service.response.segmented_rgb_images;
    
    sensor_msgs::Image::ConstPtr constptr_segmented_image(new sensor_msgs::Image(*segmented_images.begin()));
    readImage(constptr_segmented_image, mat_rgb_segmented_image);

    cv::cvtColor(mat_rgb_segmented_image, mat_grayscale_segmented_image, CV_RGB2GRAY);
    extractFeatures(descriptors[actual_iterator]);
	first_keypoint = keypoints;    
	first_descriptors = descriptors[actual_iterator];	
	actual_iterator++;
    queue_actual_size++; 
	mat_grayscale_segmented_image.copyTo(keypoints_images);
	mat_grayscale_segmented_image.copyTo(debug);
	//cv:drawKeypoints(mat_grayscale_segmented_image, first_keypoint, debug, cv::Scalar(0,255,0),cv::DrawMatchesFlags::DEFAULT );
	//cv::imshow("Features original",debug);
    initialized = true;
    res.started = true;
    return true;
}


bool PeopleTracker::stopTracking(butia_vision_msgs::StopTracking::Request &req, butia_vision_msgs::StopTracking::Response &res) {
    if (req.stop) {
        initialized = false;
        res.stopped = true;
        return true;
    }

    initialized = true;
    res.stopped = false;
    return false;
}


void PeopleTracker::peopleDetectionCallBack(const butia_vision_msgs::Recognitions::ConstPtr &person_detected) {
    butia_vision_msgs::ImageRequest image_request_service;
    butia_vision_msgs::SegmentationRequest image_segmentation_service;

    ROS_WARN("Person Detected seq: %d", person_detected->image_header.seq);

    frame_id = person_detected->image_header.seq;
    descriptions = person_detected->descriptions;
    message.header = person_detected->header;
    message.image_header = person_detected->image_header;
    if (initialized == true) {
        image_request_service.request.seq = frame_id;
        if (!image_request_client.call(image_request_service))
            ROS_ERROR("Failed to call image request service!");
        else {
            ROS_INFO("Image request service called!");

            butia_vision_msgs::RGBDImage rgbd_image = image_request_service.response.rgbd_image;

            std::vector<butia_vision_msgs::Description>::iterator it_descriptions;
            std::vector<sensor_msgs::Image>::iterator it_images;

            std::vector<sensor_msgs::Image> segmented_images;

            for (it_descriptions = descriptions.begin(); it_descriptions != descriptions.end(); it_descriptions++) {
                if (((*it_descriptions).bounding_box.width * (*it_descriptions).bounding_box.height < bounding_box_size_threshold * image_size) || ((*it_descriptions).probability < probability_threshold)) {
                    descriptions.erase(it_descriptions);
                    it_descriptions--;
                }
            }

            image_segmentation_service.request.model_id = "median_full";
            image_segmentation_service.request.descriptions = descriptions;
            image_segmentation_service.request.initial_rgbd_image = rgbd_image;
            if (!image_segmentation_client.call(image_segmentation_service))
                ROS_ERROR("Failed to call image segmentation service!");
            else {
                ROS_INFO("Image segmentation service called!");
                segmented_images = image_segmentation_service.response.segmented_rgb_images;
                
                number_of_matches_on_better_match = 0;
                person_founded = false;
                for (it_images = segmented_images.begin(), it_descriptions = descriptions.begin(); it_images != segmented_images.end(); it_images++, it_descriptions++) {
                    sensor_msgs::Image::ConstPtr constptr_segmented_image(new sensor_msgs::Image(*it_images));
                    readImage(constptr_segmented_image, mat_rgb_segmented_image);

                    cv::cvtColor(mat_rgb_segmented_image, mat_grayscale_segmented_image, CV_RGB2GRAY);

                    extractFeatures(actual_descriptors);
                    for (int i = 0; i < queue_actual_size; i++) {
                        if ((matchFeatures(descriptors[i])) && (good_matches > number_of_matches_on_better_match)) {
                            actual_better_segmented_image = mat_rgb_segmented_image;
                            number_of_matches_on_better_match = good_matches;
                            better_descriptors = actual_descriptors;
                            better_bounding_box = (*it_descriptions).bounding_box;
                            better_probability = (*it_descriptions).probability;
                            person_founded = true;
                        }
                    }
                }

                if (person_founded == true) {
                    //registerMatch();

                    butia_vision_msgs::Description desc;
                    desc.label_class = "person";
                    desc.bounding_box = better_bounding_box;
                    desc.probability = better_probability;

                    message.descriptions.push_back(desc);

                    people_tracking_publisher.publish(message);
					
					cv::Rect roi;
					roi.x = 0;
					roi.y = 0;
					roi.width = (desc.bounding_box.width);
					roi.height = (desc.bounding_box.height);

					cv::Mat crop = actual_better_segmented_image(roi); 
					cv::imshow("crop", crop);

                }
            }
        }
    } else {
        butia_vision_msgs::StartTracking::Response start_tracking_response;
        butia_vision_msgs::StartTracking::Request start_tracking_request;
        start_tracking_request.start = true;
        startTracking(start_tracking_request, start_tracking_response);
    }
}


void PeopleTracker::readImage(const sensor_msgs::Image::ConstPtr &source, cv::Mat &destiny) {
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(source, source->encoding);
    cv_image->image.copyTo(destiny);
}


void PeopleTracker::extractFeatures(cv::Mat_<float> &destiny) {
    keypoints.clear();
    destiny = cv::Mat();

    if (param_detector_type == "surf")
        surf_detector->detectAndCompute(mat_grayscale_segmented_image, cv::Mat(), keypoints, destiny);
    else if (param_detector_type == "sift")
        sift_detector->detectAndCompute(mat_grayscale_segmented_image, cv::Mat(), keypoints, destiny);
}

bool compdmatch(cv::DMatch &a, cv::DMatch &b){
	return a.distance<b.distance;
}

bool PeopleTracker::matchFeatures(cv::Mat_<float> &destiny) {
    matches.clear();
    good_matches = 0;

    matcher.match(first_descriptors, actual_descriptors, matches);
	std::sort(matches.begin(),matches.end(),compdmatch);

    float minimal_distance = matches[0].distance;
    /*for(int i = 0; i < first_descriptors.rows; i++) {
        double distance = matches[i].distance;
        if(distance < minimal_distance)
            minimal_distance = distance;
    }*/

    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= std::max(2*minimal_distance, minimal_minimal_distance))
           good_matches++;
    }

	int intermed = (matches.size()/4);
	std::vector<cv::DMatch> new_matches(matches.begin(),matches.begin()+intermed);

	for (int i = 0; i < new_matches.size(); i++) {
		std::cout << new_matches[i].distance << std::endl;
    }

	cv::Mat debug2;	
	cv::Mat img_matches;
	cv::drawMatches(keypoints_images, first_keypoint, mat_grayscale_segmented_image, keypoints, new_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	mat_grayscale_segmented_image.copyTo(debug2);
	cv:drawKeypoints(mat_grayscale_segmented_image, keypoints, debug2, cv::Scalar(0,255,0),cv::DrawMatchesFlags::DEFAULT );
	//cv::imshow("Features",debug2);	
	cv::imshow("You have a new match",img_matches);
	
	cv::waitKey(1);
	float p = (((float)good_matches)/((float)matches.size()));
	std::cout << "Minimal distance:  " << minimal_distance << std::endl;
	std::cout << "Número de keypoints - first:  " << first_keypoint.size() << std::endl;
	std::cout << "Número de keypoints - compare:  " << keypoints.size() << std::endl; 
	std::cout << "Tamanho do vetor de matches:  " << matches.size() << std::endl;	
	//std::cout << "Destiny.rows:  " << destiny.rows << std::endl;
	std::cout << "Actual.rows:  " << actual_descriptors.rows << std::endl;
	std::cout << "matches_check_factor:  " << p << std::endl;
	std::cout << "Numero de good matches:  " << good_matches << std::endl;
	std::cout << "mult:  " << (std::min(actual_descriptors.rows,first_descriptors.rows)*p)<< std::endl;
	
    if (good_matches > (std::min(actual_descriptors.rows,first_descriptors.rows)*p))
        return false;

    return true;
}


void PeopleTracker::registerMatch() {
    descriptors[actual_iterator] = better_descriptors;
    actual_iterator++;
    if (actual_iterator == queue_size)
        actual_iterator = 0;
    if (queue_actual_size < queue_size)
        queue_actual_size++;
}


void PeopleTracker::readParameters() {
    node_handle.param("/people_tracking/thresholds/bounding_box_size", bounding_box_size_threshold, (float)(0.1));
    node_handle.param("/people_tracking/thresholds/probability", probability_threshold, (float)(0.5));
    
    node_handle.param("/people_tracking/queue/size", queue_size, (int)(20));

    node_handle.param("/people_tracking/match/minimal_hessian", min_hessian, (int)(400));
    node_handle.param("/people_tracking/match/minimal_minimal_distance", minimal_minimal_distance, (float)(0.2));
    node_handle.param("/people_tracking/match/check_factor", matches_check_factor, (float)(0.5));
    node_handle.param("/people_tracking/match/k", param_k, (int)(8));

    node_handle.param("/people_tracking/detector/type", param_detector_type, std::string("surf"));

    node_handle.param("/services/people_tracking/start_tracking", param_start_service, std::string("/butia_vision/pt/start"));
    node_handle.param("/services/people_tracking/stop_tracking", param_stop_service, std::string("/butia_vision/pt/stop"));
    node_handle.param("/services/image_server/image_request", param_image_request_service, std::string("/butia_vision/is/image_request"));
    node_handle.param("/services/segmentation/segmentation_request", param_segmentation_request_service, std::string("/butia_vision/seg/image_segmentation"));

    node_handle.param("/topics/object_recognition/people_detection", param_people_detection_topic, std::string("/butia_vision/or/people_detection"));
    node_handle.param("/topics/people_tracking/people_tracking", param_people_tracking_topic, std::string("/butia_vision/pt/people_tracking"));
}
//----------------------------------------------------------------------------------------------------
