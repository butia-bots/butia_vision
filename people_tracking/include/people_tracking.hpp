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




class PeopleTracker {
    private:
        ros::NodeHandle node_handle; //Internal NodeHandle
        ros::Subscriber people_detection_subscriber; //Subscriber that reads the people detection topic
        ros::ServiceClient image_client; //Service that gets the RGBD image


    public:
        //Constructors
        PeopleTracker(ros::NodeHandle _nh);

        //CallBackers
        void peopleDetectionCallBack(const vision_system_msgs::Recognitions::ConstPtr& person_detected);

        //Image Reader
        void readImage(const sensor_msgs::Image::ConstPtr& msg_image, cv::Mat &image);
};





























//cv::Mat getMask(const cv::Mat depth_image, int bounding_box_size); //Create the mask based on the depth histogram