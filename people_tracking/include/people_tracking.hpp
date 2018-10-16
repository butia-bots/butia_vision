#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv.h>
#include <vector>
#include <string>
#include <utility>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/BoundingBox.h"
#include "vision_system_msgs/RGBDImage.h"
#include "vision_system_msgs/Recognitions.h"




class PeopleTracker {
    private:
        vision_system_msgs::ImageRequest srv; //Service that requests the rgbd images from the image server


    public:
        //Callbacks
        void peopleDetectCallback(const vision_system_msgs::RecognitionsConstPtr person);

        //Functions that manipulate images
        std::pair<cv::Mat, cv::Mat> crop(vision_system_msgs::RGBDImage rgbd_image, vision_system_msgs::BoundingBox bounding_box); //Crop the images
        cv::Mat getMask(const cv::Mat depth_image, int bounding_box_size); //Create the mask based on the depth histogram
        cv::Mat segment(cv::Mat rgb_image, cv::Mat mask); //Apply the mask on the rgb image
        int getMax(cv::Mat histogram); //Get the maximum value of the histogram*/
};