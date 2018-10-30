# image_server
## Overview
The image_server reads images and camera info syncronized from vision_system_bridge and stores them in buffers to enable requisitions of an especific seq in a service. 
## Subscribers
* **`image_rgb:`** `/vision_system/vsb/image_rgb_raw` ([sensor_msgs::Image])

    Color images from vision_system_bridge.
    
* **`image_depth:`** `/vision_system/vsb/image_depth_raw` ([sensor_msgs::Image])

    Depth images from vision_system_bridge.

* **`camera_info:`** `/vision_system/vsb/camera_info` ([sensor_msgs::CameraInfo])
    
    Camera infomation from vision_system_bridge.

## Servers
* **`image_request:`** `/vision_system/is/image_request` ([vision_system_msgs::ImageRequest])
    
    Service that provides a RGBDImage and a CameraInfo for a specific seq.

## Nodes
* **`image_server`**

    Reads vision_system_bridge topics syncronized, stores on buffers provides for access by service.