#include "ros/ros.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

#include "darknet_ros_msgs/BoundingBoxes.h"

#include "butia_vision_msgs/Recognitions.h"
#include "butia_vision_msgs/ListClasses.h"

std::vector<std::string> DEFAULT_CLASS_LIST = {"person", "bicycle", "bench", "backpack", "umbrella", "handbag", "suitcase", "sports ball",
                                               "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
                                               "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
                                               "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", 
                                               "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", 
                                               "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

class YoloRecognition{
    public:
        YoloRecognition(ros::NodeHandle _nh);

        bool getObjectList(butia_vision_msgs::ListClasses::Request &req, butia_vision_msgs::ListClasses::Response &res);

        void yoloRecognitionCallback(darknet_ros_msgs::BoundingBoxes bbs);

    private:
        ros::NodeHandle node_handle;

        ros::Subscriber bounding_boxes_sub;

        butia_vision_msgs::Recognitions pub_object_msg;
        butia_vision_msgs::Recognitions pub_people_msg;

        ros::Publisher recognized_objects_pub;
        ros::Publisher recognized_people_pub;
        ros::Publisher object_list_updated_pub;

        ros::ServiceServer list_objects_server;
        
        std::string bounding_boxes_topic;
        int bounding_boxes_qs;

        std::string object_recognition_topic;
        int object_recognition_qs;

        std::string people_detection_topic;
        int people_detection_qs;

        std::string object_list_updated_topic;
        int object_list_updated_qs;

        std::string list_objects_service;

        std::string person_identifier;

        float threshold;

        std::vector<std::string> possible_classes;

        std_msgs::Header object_list_updated_header;

        void readParameters();
};
