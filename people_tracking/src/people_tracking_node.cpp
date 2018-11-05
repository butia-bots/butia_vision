#include "people_tracking.hpp"



int main(int argc, char **argv) {
    ros::init(argc, argv, "people_tracking_node");

    ros::NodeHandle node_handle;

    PeopleTracker people_tracker(node_handle);

    ros::spin();
    
    return 0;
}