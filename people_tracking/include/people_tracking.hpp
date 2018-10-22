#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <utility>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/BoundingBox.h"
#include "vision_system_msgs/RGBDImage.h"
#include "vision_system_msgs/Recognitions.h"
#include "vision_system_msgs/Description.h"
#include "vision_system_msgs/ImageSegmentation.h"




class PeopleTracker {
    private:
        ros::NodeHandle node_handle; //Internal NodeHandle
        ros::Subscriber people_detection_subscriber; //Subscriber that reads the people detection topic
        ros::ServiceClient image_request_client; //Service that gets the RGBD image
        ros::ServiceClient image_segmentation_client; //Service that segments the person

        int image_size;
        float bounding_box_size_threshold;


    public:
        PeopleTracker(ros::NodeHandle _nh); //Constructor

        void peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr &person_detected); //CallBack
};



