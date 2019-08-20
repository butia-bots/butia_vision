#include "butia_vision_bridge/butia_vision_bridge.h"
#include <tf/transform_broadcaster.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "butia_vision_bridge_node");
    ros::NodeHandle nh;

    ButiaVisionBridge butia_vision_bridge(nh);

    ros::Rate rate(30);

    std::string robot_tf, kinect_tf;
    double offset_x, offset_y, offset_z;

    nh.param("/butia_vision_bridge/robot_tf", robot_tf, std::string("base_link"));
    nh.param("/butia_vision_bridge/kinect_tf", kinect_tf, std::string("kinect2_link"));
    nh.param("/butia_vision_bridge/offset_x", offset_x, 0.1315);
    nh.param("/butia_vision_bridge/offset_y", offset_y, 0.0020);
    nh.param("/butia_vision_bridge/offset_z", offset_z, 1.2120);
    tf::Vector3 vec(offset_x, offset_y, offset_z);

    tf::TransformBroadcaster broadcaster;

    while(nh.ok()) {
        broadcaster.sendTransform(tf::StampedTransform( tf::Transform(tf::Quaternion(0, 0, 0, 1), vec), ros::Time::now(), robot_tf, kinect_tf));
        ros::spinOnce();
        rate.sleep();
    } 
}
