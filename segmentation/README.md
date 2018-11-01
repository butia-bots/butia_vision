# people_tracking
## Overview
The segmentation package segmentate the rgb image based on the histogram of the depth image.

### Services
* **`imamge_segmentation`**

The image_segmentation service receives the bounding box and the rgbd image as requests. Based in this informations, the service segments the rgb image and send it as response.

### Nodes
* **`segmentation_node`**

    Tracking at real-time.