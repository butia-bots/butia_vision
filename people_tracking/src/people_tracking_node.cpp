#include <ros/ros.h>
#include <opencv2/opencv.hpp>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/RecognizedPeople.h"
#include "vision_system_msgs/ImageRequest.h"





void peopleRecoCallBack(const vision_system_msgs::RecognizedPeopleConstPtr reco) {

}



int main(int argc, char **argv) {
    ros::init(argc, argv, "people_tracking_node");
    ros::NodeHandle nh;
    ros::Subscriber reco_people_sub = nh.subscribe("/vision_system/or/people_recognition", 100, &peopleRecoCallBack);
    ros::ServiceClient img_server_client = nh.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/img_server/image_request");
    vision_system_msgs::ImageRequest srv;
    srv.response.image;
}