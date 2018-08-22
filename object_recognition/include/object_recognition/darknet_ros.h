#include <ros.h>

#include "darknet_ros_msgs/BoundingBoxes.h"
#include "vision_system_msgs/RecognizedObjects.h"

//A class that will set the parameters in rosparam server and make a interface of object_recognition and darknet_ros packages

class DarknetROS{
    public:
        DarknetROS(ros::NodeHandle _nh);

        void darknetRosCallback();
        void objectRecognitionPublish();

    private:
        ros::NodeHandle nh;
        ros::Publisher  recognized_objects_pub;
        ros::Subscriber bounding_boxes_sub;
        
        void load();
        void setParameters();
};