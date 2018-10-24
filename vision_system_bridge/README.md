# img_server
## Overview
The image_server package is an internal image server to support the system to get the images in real time.
## Services
* **`image_request:`** `/vision_system/img_server/image_request` ([vision_system_msgs::ImageRequest])

    Ask for a frame from the image server.
## Nodes
* **`img_server_node`**

    Updates the internal image server at real time.