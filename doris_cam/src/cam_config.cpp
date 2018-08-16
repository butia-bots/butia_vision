#include "doris_cam/includes/cam_config.hpp"

DorisCam::DorisCam() {
    cam_config.width = 640;
    cam_config.height = 480;
    cam_config.bytes_per_frame = 0.375;
}

DorisCam::DorisCam(uint larg, uint alt, float bpf) {
    cam_config.width = larg;
    cam_config.height = alt;
    cam_config.bytes_per_frame = bpf;
}

DorisCam::~DorisCam() {
    stopRecording();
}