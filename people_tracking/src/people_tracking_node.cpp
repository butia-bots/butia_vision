#include "people_tracking.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "people_tracking_node");
    ros::NodeHandle nh;

    ImagePreparer image_preparer;
    ros::Subscriber people_recognition_sub = nh.subscribe("/vision_system/or/people_recognition", 75, &ImagePreparer::peopleRecoCallBack, &image_preparer);
    ros::ServiceClient img_server_client = nh.serviceClient<vision_system_msgs::ImageRequest>("/vision_system/img_server/image_request");
    
    ros::spin();
}