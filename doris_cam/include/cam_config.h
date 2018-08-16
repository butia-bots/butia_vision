#ifndef CAM_CONFIG_H
#define CAM_CONFIG_H


#include <sstream>
#include <opencv2/opencv.hpp>



typedef struct {
    uint width;
    uint height;
    uint bytes_per_frame;
    uint frames_per_second;
} Image_Config;

typedef struct {
    uint size;
    std::queue<cv::Mat> img_queue;
} Buffer;



class DorisCam {
    private:
        Image_Config cam_config;

    public:
        Buffer img_server;
        cv::VideoCapture cap;

        DorisCam();
        DorisCam(uint larg, uint alt, uint fps);
        int startRecording();
        int stopRecording();
        bool isRecording();
};


#endif