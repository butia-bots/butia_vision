#include "people_tracking.hpp"



//------------------------------People Tracker's Functions------------------------------
void PeopleTracker::peopleDetectCallback(const vision_system_msgs::RecognitionsConstPtr person) {
    srv.request.frame = person->image_header.seq; //Getting the frame
    std::pair<cv::Mat, cv::Mat> images; //This variable stores the rgb and the depth image, in this order
    cv::Mat mask, segmented_image; //This variable stores the segmented rgb image

    //Treating all the people in the frame
    for (int i = 0; i < person->descriptions.size(); i++) {
        images = crop(srv.response.rgbd_image, person->descriptions[i].bounding_box); //Cropping the image to the size of the bounding box
        mask = getMask(images.second, person->descriptions[i].bounding_box.width * person->descriptions[i].bounding_box.height); //Getting the masks
        segmented_image = segment(images.first, mask); //Segmenting the images
    }
}


std::pair<cv::Mat, cv::Mat> PeopleTracker::crop(vision_system_msgs::RGBDImage rgbd_image, vision_system_msgs::BoundingBox bounding_box) {
    //Getting the images
    cv::Mat initial_rgb_image = (cv_bridge::toCvCopy(rgbd_image.rgb_image, rgbd_image.rgb_image.encoding))->image;
    cv::Mat initial_depth_image = (cv_bridge::toCvCopy(rgbd_image.depth_image, rgbd_image.depth_image.encoding))->image;

    //Defining the region of interest
    cv::Rect region_of_interest(bounding_box.minX, bounding_box.minY, bounding_box.width, bounding_box.height);

    //Cutting the images
    std::pair<cv::Mat, cv::Mat> final_images;
    final_images.first = initial_rgb_image(region_of_interest);
    final_images.second = initial_depth_image(region_of_interest);
    return final_images;
}


cv::Mat PeopleTracker::getMask(const cv::Mat depth_image, int bounding_box_size) {
    //Defining parameters
    cv::Mat histogram;
    int histogram_size = 64;
    float range[] = {0, 5001};
    const float *histogram_range = {range};
    bool accumulate = false;
    bool uniform = true;
    const int channels[] = {0};
    float bounding_box_threshold = 0.65;

    //Calculating the histograms until the max value be consistent with the bouding box
    int number_of_pixels;
    do {
        cv::calcHist(&depth_image, 1, channels, cv::Mat(), histogram, 1, &histogram_size, &histogram_range, uniform, accumulate);
        number_of_pixels = getMax(histogram);
        if (number_of_pixels < bounding_box_size * bounding_box_threshold)
            histogram_size = histogram_size / 1.5;
    } while (number_of_pixels < bounding_box_size * bounding_box_threshold);
    int reference_value = histogram.at<int>((int)(histogram_range[1] / histogram_size), 0) * number_of_pixels; //Defining the reference depth value
    int mask_threshold = (histogram_range[1] / histogram_size) / 2; //Defining the threshold

    //Creating the mask with the value 0 in the background pixels and value 1 int the foreground pixels
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
}


cv::Mat PeopleTracker::segment(cv::Mat rgb_image, cv::Mat mask) {
    cv::Mat segmented_image;

    //Multiplying the matrices pixel-by-pixel
    for (int i = 0; i < rgb_image.rows; i++) {
        for (int j = 0; j < rgb_image.cols; j++)
            segmented_image.at<int>(i, j) = rgb_image.at<int>(i, j) * mask.at<int>(i, j);
    }

    return segmented_image;
}


int PeopleTracker::getMax(cv::Mat histogram) {
    int max = 0; //Set the initial maximum value (zero for any value be bigger)

    //Scrolls the vector looking for the bigger value
    for (int i = 0; i < histogram.cols; i++) {
        if (max < histogram.at<int>(i, 0))
            max = histogram.at<int>(i, 0);
    }

    return max;
}
//------------------------------People Tracker's Functions------------------------------