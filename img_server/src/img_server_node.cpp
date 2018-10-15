#include "img_server.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "img_server_node");
    ros::NodeHandle nh;
    ImgServer img_server(150);

    ros::Subscriber img_rgb_sub = nh.subscribe("/kinect2/qhd/image_color_rect", 100, &ImgServer::camCallbackRGB, &img_server);
    ros::Subscriber img_d_sub = nh.subscribe("/kinect2/qhd/image_depth_rect", 100, &ImgServer::camCallbackDepth, &img_server);
    ros::ServiceServer service = nh.advertiseService("/vision_system/is/image_request", &ImgServer::accessQueue, &img_server);
    ros::spin();

    cv::destroyAllWindows();
}