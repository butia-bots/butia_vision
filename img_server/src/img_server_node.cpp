#include "img_server.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "img_server_node");
    ros::NodeHandle nh;
    ImgServer img_server;

    ros::Subscriber img_rgb_sub = nh.subscribe("/rgb/image_raw", 100, &ImgServer::camCallBackRGB, &img_server);
    ros::Subscriber img_d_sub = nh.subscribe("/depth/image_raw", 100, &ImgServer::camCallBackDepth, &img_server);
    ros::ServiceServer service = nh.advertiseService("/vision_system/img_server/image_request", &ImgServer::accessQueue, &img_server);
    ros::spin();
}