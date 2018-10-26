#include "image_server/image_server.h"


int main(int argc, char **argv) {
    ros::init(argc, argv, "image_server_node");
    ros::NodeHandle nh;

    ImageServer image_server(nh);

    ros::spin();
}