#include "vision_system_bridge.h"


int main(int argc, char **argv) {
    ros::init(argc, argv, "vision_system_bridge_node");
    ros::NodeHandle nh;

    VisionSystemBridge vision_system_bridge(nh);

    
    ros::spin();
}