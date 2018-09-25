#include "people_tracking.hpp"



//------------------------------Image Prepare's Functions------------------------------
void ImagePreparer::peopleRecoCallBack(const vision_system_msgs::RecognizedPeopleConstPtr person) {
    srv.request.frame = person->image_header.seq; //Getting the frame

    //Treating all the people in the frame
    for (int i = 0; i < person->people_description.size(); i++)
        crop(srv.response.rgbd_image, person->people_description[i].bounding_box); //Crop the image to the size of the bounding box
    local_server.clear(); //Clean the local server
}


void ImagePreparer::crop(vision_system_msgs::RGBDImage rgbd_image, vision_system_msgs::BoundingBox bounding_box) {
    //Getting the images
    cv::Mat initial_rgb_image = (cv_bridge::toCvCopy(rgbd_image.rgb, rgbd_image.rgb.encoding))->image;
    cv::Mat initial_depth_image = (cv_bridge::toCvCopy(rgbd_image.depth, rgbd_image.depth.encoding))->image;

    //Defining the region of interest
    cv::Rect roi;
    roi.x = bounding_box.width;
    roi.y = bounding_box.height;
    roi.width = initial_rgb_image.size().width - (roi.x * 2);
    roi.height = initial_rgb_image.size().height - (roi.y * 2);

    //Putting on the local image server
    std::pair<cv::Mat, cv::Mat> final_image;
    final_image.first = initial_rgb_image(roi);
    final_image.second = initial_depth_image(roi);
    local_server.push_back(final_image);
}
//------------------------------Image Prepare's Functions------------------------------