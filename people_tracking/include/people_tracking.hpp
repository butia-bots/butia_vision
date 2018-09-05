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




class ImagePreparer {
    private:
        vision_system_msgs::ImageRequest srv;
        std::vector<cv::Mat> local_server;
        int imagenum;

    public:
        ImagePreparer() {imagenum = 0;}

        void peopleRecoCallBack(const vision_system_msgs::RecognizedPeopleConstPtr person) {
            srv.request.frame = person->image_header.seq;
            for (int i = 0; i < person->people_description.size(); i++)
                crop(srv.response.image, person->people_description[i].bounding_box);
        }


        void crop(sensor_msgs::Image image, vision_system_msgs::BoundingBox bounding_box) {
            cv::Mat initial_image = (cv_bridge::toCvCopy(image, image.encoding))->image;

            cv::Rect roi;
            roi.x = bounding_box.width;
            roi.y = bounding_box.height;
            roi.width = initial_image.size().width - (roi.x * 2);
            roi.height = initial_image.size().height - (roi.y * 2);

            cv::Mat final_image = initial_image(roi);
            local_server.push_back(final_image);

            std::string image_name = std::to_string(imagenum) + ".pgm";
            cv::imwrite(image_name, final_image);

            imagenum++;
        }
};