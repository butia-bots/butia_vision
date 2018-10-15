#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/ImageRequest.h"
#include "vision_system_msgs/RGBDImage.h"



class ImgServer {
    private:
        std::vector<std::pair<sensor_msgs::Image::ConstPtr, sensor_msgs::Image::ConstPtr>> buffer;
        

    public:
        //Constructors
        ImgServer(int size);

        //Server
        bool accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res);

        //Callbacks
        void camCallbackRGB(const sensor_msgs::Image::ConstPtr& img);
        void camCallbackDepth(const sensor_msgs::Image::ConstPtr& img);
};