#include "people_tracking.hpp"




//------------------------------People Tracker's Functions------------------------------
void PeopleTracker::peopleRecoCallBack(const vision_system_msgs::RecognizedPeopleConstPtr person) {
    srv.request.frame = person->image_header.seq; //Getting the frame
    std::pair<cv::Mat, cv::Mat> images; //This variable stores the rgb and the depth image, in this order

    //Treating all the people in the frame
    for (int i = 0; i < person->people_description.size(); i++) {
        images = crop(srv.response.rgbd_image, person->people_description[i].bounding_box); //Crop the image to the size of the bounding box
    }
}



std::pair<cv::Mat, cv::Mat> PeopleTracker::crop(vision_system_msgs::RGBDImage rgbd_image, vision_system_msgs::BoundingBox bounding_box) {
    //Getting the images
    cv::Mat initial_rgb_image = (cv_bridge::toCvCopy(rgbd_image.rgb, rgbd_image.rgb.encoding))->image;
    cv::Mat initial_depth_image = (cv_bridge::toCvCopy(rgbd_image.depth, rgbd_image.depth.encoding))->image;

    //Defining the region of interest
    cv::Rect roi(bounding_box.minX, bounding_box.minY, bounding_box.width, bounding_box.height);

    //Putting on the local image server
    std::pair<cv::Mat, cv::Mat> final_image;
    final_image.first = initial_rgb_image(roi);
    final_image.second = initial_depth_image(roi);
    return final_image;
}
//------------------------------People Tracker's Functions------------------------------