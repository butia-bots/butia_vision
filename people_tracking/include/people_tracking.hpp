#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/RecognizedPeople.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/BoundingBox.h"
#include "vision_system_msgs/RGBDImage.h"






class ImagePreparer {
    private:
        vision_system_msgs::ImageRequest srv;
        std::vector<cv::Mat> local_server_rgb;
        std::vector<cv::Mat> local_server_depth;
        int imagenum;



    public:
        //Constructor
        ImagePreparer() {imagenum = 0;}


        //Callback
        void peopleRecoCallBack(const vision_system_msgs::RecognizedPeopleConstPtr person) {
            srv.request.frame = person->image_header.seq;
            for (int i = 0; i < person->people_description.size(); i++)
                crop(srv.response.rgbd_image, person->people_description[i].bounding_box);
        }


        //Cropper
        void crop(vision_system_msgs::RGBDImage rgbd_image, vision_system_msgs::BoundingBox bounding_box) {
            cv::Mat initial_rgb_image = (cv_bridge::toCvCopy(rgbd_image.rgb, rgbd_image.rgb.encoding))->image;
            cv::Mat initial_depth_image = (cv_bridge::toCvCopy(rgbd_image.depth, rgbd_image.depth.encoding))->image;

            cv::Rect rgb_roi;
            cv::Rect depth_roi;
            rgb_roi.x = bounding_box.width;
            depth_roi.x = bounding_box.width;
            rgb_roi.y = bounding_box.height;
            depth_roi.y = bounding_box.height;
            rgb_roi.width = initial_rgb_image.size().width - (rgb_roi.x * 2);
            depth_roi.width = initial_depth_image.size().width - (depth_roi.x * 2);
            rgb_roi.height = initial_rgb_image.size().height - (rgb_roi.y * 2);
            depth_roi.height = initial_depth_image.size().height - (depth_roi.y * 2);

            cv::Mat final_rgb_image = initial_rgb_image(rgb_roi);
            cv::Mat final_depth_image = initial_depth_image(depth_roi);
            local_server_rgb.push_back(final_rgb_image);
            local_server_depth.push_back(final_depth_image);

            std::string rgb_image_name = std::to_string(imagenum) + "_rgb.png";
            std::string depth_image_name = std::to_string(imagenum) + "_depth.png";
            cv::imwrite(rgb_image_name, final_rgb_image);
            cv::imwrite(depth_image_name, final_depth_image);

            imagenum++;
        }
};