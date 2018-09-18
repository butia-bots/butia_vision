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
        //Constructor
        ImgServer() {buffer.resize(150);}
        ImgServer(int size) {buffer.resize(size);}

        //Getter
        std::vector<std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>> getBuffer() {return buffer;}

        //Server
        bool accessQueue(vision_system_msgs::ImageRequest::Request &req, vision_system_msgs::ImageRequest::Response &res) {
            res.rgbd_image.rgb = *(buffer[(req.frame)%150].first);
            res.rgbd_image.depth = *(buffer[(req.frame)%150].second);
            return true;
        }

        //Callbacks
        void camCallBackRGB(const sensor_msgs::ImageConstPtr img) {buffer[(img->header.seq)%150].first = img;}
        void camCallBackDepth(const sensor_msgs::ImageConstPtr img) {buffer[(img->header.seq)%150].second = img;}
};