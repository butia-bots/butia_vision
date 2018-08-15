#include <sstream>
#include <opencv2/opencv.hpp>



typedef struct {
    uint width;
    uint height;
    int bytes_per_frame;
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

        DorisCam();
        DorisCam(uint larg, uint alt, int bpf);
        ~DorisCam();
        int startRecording();
        int stopRecording();
        bool isRecording();
};