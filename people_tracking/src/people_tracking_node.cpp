#include "people_tracking.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "people_tracking_node");
    ros::NodeHandle nh;

    PeopleTracker people_tracker;
    ros::Subscriber people_recognition_sub = nh.subscribe("/vision_system/or/people_recognition", 75, &PeopleTracker::peopleRecoCallBack, &people_tracker);
    ros::ServiceClient img_server_client = nh.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/img_server/image_request");
    
    ros::spin();
}