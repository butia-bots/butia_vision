#include "doris_cam/includes/cam_config.hpp"

DorisCam::DorisCam() {
    cam_config.width = 640;
    cam_config.height = 480;
    cam_config.bytes_per_frame = 307200;
    cam_config.frames_per_second = 30;
}

DorisCam::DorisCam(uint larg, uint alt, uint fps) {
    cam_config.width = larg;
    cam_config.height = alt;
    cam_config.bytes_per_frame = larg*alt;
    cam_config.frames_per_second = fps;
}


bool DorisCam::isRecording() {
    return cap.isOpened();
}


int DorisCam::startRecording() {
    cap.open(0);

    if (isRecording())
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;
}


int DorisCam::stopRecording() {
    if (isRecording()) {
        cap.release();
        if (isRecording())
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}