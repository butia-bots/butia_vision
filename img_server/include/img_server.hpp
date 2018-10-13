#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "vision_system_msgs/ImageRequest.h"



class ImgServer {
    private:
        std::vector<std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>> buffer;
        

    public:
        //Constructors
        ImgServer();
        ImgServer(int size);

        //Getter
        std::vector<std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>> getBuffer();

        //Server
        bool accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res);

        //Callbacks
        void camCallBackRGB(const sensor_msgs::ImageConstPtr img);
        void camCallBackDepth(const sensor_msgs::ImageConstPtr img);
};