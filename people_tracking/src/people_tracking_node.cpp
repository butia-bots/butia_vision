#include "people_tracking.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "people_tracking_node");
    ros::NodeHandle nh;

    PeopleTracker people_tracker;
    ros::ServiceClient img_server_client = nh.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/is/image_request");

    ros::Subscriber img_rgb_sub = nh.subscribe("/kinect2/qhd/image_color_rect", 100, &PeopleTracker::camCallbackRGB, &people_tracker);
    ros::Subscriber img_d_sub = nh.subscribe("/kinect2/qhd/image_depth_rect", 100, &PeopleTracker::camCallbackDepth, &people_tracker);
    ros::Subscriber people_detection_sub = nh.subscribe("/vision_system/or/people_detection", 75, &PeopleTracker::peopleDetectCallback, &people_tracker);

    ros::spin();
}