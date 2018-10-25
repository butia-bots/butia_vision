# vision_system_bridge
## Overview
The vision_system_bridge package is used to read images from kinect_bridge and publish the syncronized. Besides that, the package provides a service for request a specific image pair (rgb and depth) and camera_info.
## Subscribers
* **`image_rgb:`** `/kinect2/qhd/image_color_rect` ([sensor_msgs::Image])

    Color images from kinect_bridge.
    
* **`image_depth:`** `/kinect2/qhd/image_depth_rect` ([sensor_msgs::Image])

    Depth images from kinect_bridge.

* **`camera_info:`** `/kinect2/qhd/camera_info` ([sensor_msgs::CameraInfo])
    
    Camera infomation from kinect_bridge.

## Publishers
* **`image_rgb:`** `/vision_system/vsb/image_rgb_raw` ([sensor_msgs::Image])

    Publishs an image color that have a depth pair.
    
* **`image_depth:`** `/vision_system/vsb/image_depth_raw` ([sensor_msgs::Image])

    Publishs an image depth that have a color pair.

* **`camera_info:`** `/vision_system/vsb/camera_info` ([sensor_msgs::CameraInfo])

    Publishs camera information.

## Services
* **`image_request:`** `/vision_system/img_server/image_request` ([vision_system_msgs::ImageRequest])

    Ask for a frame from the image server.

## Nodes
* **`vision_system_bridge`**

    Reads kinect_bridge topics syncronized, stores on buffers and publishs to the rest of the system.