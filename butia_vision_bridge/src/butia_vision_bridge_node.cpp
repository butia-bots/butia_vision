#include "butia_vision_bridge/butia_vision_bridge.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "butia_vision_bridge_node");
    ros::NodeHandle nh;

    ros::AsyncSpinner spinner(2);
    spinner.start();

    ButiaVisionBridge butia_vision_bridge(nh);

    ros::waitForShutdown();
}
