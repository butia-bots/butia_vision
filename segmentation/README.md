# people_tracking
## Overview
The segmentation package segmentate the rgb image based on the histogram, median of the full image or median of the center of the depth image.

## Servers
* **`image_segmentation`** `/vision_system/seg/image_segmentation` ([vision_system_msgs::SegmentationRequest])

The image_segmentation service receives the bounding box and the rgbd image as requests. Based in this informations, the service segments the rgb image and send it as response.

### Nodes
* **`segmentation_node`**

    Segmentation when asked for.