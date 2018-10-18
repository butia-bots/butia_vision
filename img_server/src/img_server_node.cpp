#include "img_server.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "img_server_node");
    ros::NodeHandle nh;

    ImgServer img_server(nh);

    
    ros::spin();
}