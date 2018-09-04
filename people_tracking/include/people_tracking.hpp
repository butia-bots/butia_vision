#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/RecognizedPeople.h"
#include "vision_system_msgs/ImageRequest.h"




class ImagePreparer {
    private:
        ros::Subscriber reco_people_sub;
        ros::ServiceClient img_server_client;
        vision_system_msgs::ImageRequest srv;
        std::vector<sensor_msgs::Image> local_buffer;

    public:
        ImagePreparer(ros::NodeHandle nh) {
            reco_people_sub = nh.subscribe("/vision_system/or/people_recognition", 100, &this->peopleRecoCallBack, this);
            img_server_client = nh.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/img_server/image_request");
        }

        void peopleRecoCallBack(const vision_system_msgs::RecognizedPeopleConstPtr person) {
            srv.request.frame = person->image_header.seq;
            local_buffer.push_back(srv.response.image);
        }
};