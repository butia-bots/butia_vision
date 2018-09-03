#include <ros.h>

#include "darknet_ros_msgs/BoundingBoxes.h"
#include "vision_system_msgs/RecognizedObjects.h"
#include "vision_system_msgs/RecognizedPeople.h"

//A class that will set the parameters in rosparam server and make a interface of object_recognition and darknet_ros packages

class DarknetROS{
    public:
        DarknetROS(ros::NodeHandle _nh);

        void darknetRosCallback(darknet_ros_msgs::BoundingBoxes bbs);

    private:
        ros::NodeHandle nh;
        ros::Publisher recognized_objects_pub;
        ros::Publisher recognized_people_pub;
        ros::Subscriber bounding_boxes_sub;
        
        void loadParameters();
};