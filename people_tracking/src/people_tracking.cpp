#include "people_tracking.hpp"



//------------------------------People Tracker's Functions------------------------------
PeopleTracker::PeopleTracker(ros::NodeHandle _nh) : node_handle(_nh), image_size(640*480), bounding_box_size_threshold(0.1) {
    people_detection_subscriber = node_handle.subscribe("/vision_system/or/people_detection", 150, &PeopleTracker::peopleDetectionCallBack, this);
    image_request_client = node_handle.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/is/image_request");
    image_segmentation_client = node_handle.serviceClient<vision_system_msgs::ImageSegmentation>("/vision_system/seg/image_segmentation");
}


void PeopleTracker::peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr &person_detected) {
    vision_system_msgs::ImageRequest image_request_service;
    vision_system_msgs::ImageSegmentation image_segmentation_service;

    image_request_service.request.frame = person_detected->image_header.seq;
    if (!image_request_client.call(image_request_service))
        ROS_ERROR("Failed to call image request service!");
    else {
        ROS_INFO("Image request service called!");

        vision_system_msgs::RGBDImage rgbd_image = image_request_service.response.rgbd_image;

        std::vector<vision_system_msgs::Description> descriptions = person_detected->descriptions;
        std::vector<vision_system_msgs::Description>::iterator it;

        for (it = descriptions.begin(); it != descriptions.end(); it++) {
            if ((*it).bounding_box.height * (*it).bounding_box.width < image_size * bounding_box_size_threshold) {
                ROS_ERROR("This bounding box is too small!");
                continue;
            }
            if ((*it).probability < 0.7) {
                ROS_ERROR("This person have a too small probability!");
                continue;
            }
            image_segmentation_service.request.bounding_box = (*it).bounding_box;
            image_segmentation_service.request.initial_rgbd_image = rgbd_image;
            if (!image_segmentation_client.call(image_segmentation_service))
                ROS_ERROR("Failed to call image segmentation service!");
            else
                ROS_INFO("Image segmentation service called!");
        }
    }
}
//------------------------------People Tracker's Functions------------------------------