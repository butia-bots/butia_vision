#include <ros/ros.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "image_transport/image_transport.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"




class  ImgServer {
    private:
        std::vector<sensor_msgs::Image> queue; //image buffer
        int size; //queue size
        ros::NodeHandle nh;
    


    public:
        //constructors and destructors
        ImgServer() {
            size = 150;
            queue.resize(size);
            image_transport::ImageTransport it(nh);
            image_transport::Subscriber img_sub = it.subscribe("/usb_cam/image_raw", 1, &ImgServer::camCallback);
        }

        ~ImgServer() {queue.clear();}


        //setters
        void setQueueSize(int new_size) {
            size = new_size;
            queue.resize(size);
        }
        

        //getters
        int getQueueSize() {return size;}

        std::string getEncodingType(int frame_number) {return queue[frame_number].encoding;}

        cv::Mat getImage(int frame_number) {
            cv_bridge::CvImagePtr img = cv_bridge::toCvCopy(queue[frame_number], "bgr8");
            return img->image;
        }


        //callback queue inserter
        void camCallback(const sensor_msgs::Image image) {queue[(image.header.seq)%size] = image;}
};