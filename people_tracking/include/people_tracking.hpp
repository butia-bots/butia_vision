#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "vision_system_msgs/RecognizedPeople.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/BoundingBox.h"
#include "vision_system_msgs/RGBDImage.h"




class PeopleTracker {
    private:
        vision_system_msgs::ImageRequest srv;


    public:
        //Callback
        void peopleRecoCallBack(const vision_system_msgs::RecognizedPeopleConstPtr person);

        //Functions that manipulate images
        void crop(vision_system_msgs::RGBDImage rgbd_image, vision_system_msgs::BoundingBox bounding_box);
};