#include "segmentation.hpp"


int main(int argc, char **argv) {
    ros::init(argc, argv, "segmentation_node");
    
    ros::NodeHandle node_handle;

    ImageSegmenter image_segmenter(node_handle);

    ros::spin();

    return 0;
}