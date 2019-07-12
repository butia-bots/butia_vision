# image_server
## Overview
The image_server reads images and camera info syncronized from butia_vision_bridge and stores them in buffers to enable requisitions of an especific seq in a service. 
## Subscribers
* **`image_rgb:`** `/butia_vision/bvb/image_rgb_raw` ([sensor_msgs::Image])

    Color images from butia_vision_bridge.
    
* **`image_depth:`** `/butia_vision/bvb/image_depth_raw` ([sensor_msgs::Image])

    Depth images from butia_vision_bridge.

* **`camera_info:`** `/butia_vision/bvb/camera_info` ([sensor_msgs::CameraInfo])
    
    Camera infomation from butia_vision_bridge.

## Servers
* **`image_request:`** `/butia_vision/is/image_request` ([butia_vision_msgs::ImageRequest])
    
    Service that provides a RGBDImage and a CameraInfo for a specific seq.

## Nodes
* **`image_server`**

    Reads butia_vision_bridge topics syncronized, stores on buffers provides for access by service.