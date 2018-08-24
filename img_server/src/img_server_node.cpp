#include "img_server_cfg.hpp"



int main(int argc, char **argv) {
    ImgServer buf;

    ros::init(argc, argv, "img_server");

    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::startWindowThread();

    int i = 0;


    while (ros::ok()) {
        cv::imshow("Image", buf.getImage(i));
        i++;
        if (i >= 150)
            i = 0;
    }

    cv::destroyAllWindows();
}