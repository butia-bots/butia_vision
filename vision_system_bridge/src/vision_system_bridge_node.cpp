#include "vision_system_bridge/vision_system_bridge.h"
#include <tf/transform_broadcaster.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "vision_system_bridge_node");
    ros::NodeHandle nh;

    VisionSystemBridge vision_system_bridge(nh);

    ros::Rate rate(30);

    std::string robot_tf, kinect_tf;
    double offset_x, offset_y, offset_z;

    nh.param("/vision_system_bridge/parameters/robot_tf", robot_tf, std::string("base_link"));
    nh.param("/vision_system_bridge/parameters/kinect_tf", kinect_tf, std::string("base_kinect2"));
    nh.param("/vision_system_bridge/parameters/offset_x", offset_x, 0.1315);
    nh.param("/vision_system_bridge/parameters/offset_y", offset_y, 0.0020);
    nh.param("/vision_system_bridge/parameters/offset_z", offset_z, 1.2120);
    tf::Vector3 vec(offset_x, offset_y, offset_z);

    tf::TransformBroadcaster broadcaster;

    while(nh.ok()) {
        broadcaster.sendTransform(tf::StampedTransform( tf::Transform(tf::Quaternion(0, 0, 0, 1), vec), ros::Time::now(), robot_tf, kinect_tf));
        ros::spinOnce();
        rate.sleep();
    }   
}