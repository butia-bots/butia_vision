# object_recognition
## Overview
The object_recognition package uses darknet_ros information and redistribute in two topics, one for objects and other for people.
## Topics
### Subscribed Topics
* **`bounding_boxes:`** `/darknet_ros/bounding_boxes` ([darknet_ros_msgs::BoundingBoxes])

    Publishes an array of bounding boxes that gives information of the position and size of the bounding box in pixel coordinates.
### Published Topics
* **`object_recognition:`** `/vision_system/or/object_recognition` ([vision_system_msgs::Recognitions])

    Publishes the recognized objects.

* **`object_recognition3d:`** `/vision_system/or/object_recognition3d` ([vision_system_msgs::Recognitions3D])

    Publishes the recognized objects in 3d.
    
* **`people_detection:`** `/vision_system/or/people_detection` ([vision_system_msgs::Recognitions])

    Publishes the recognized people.
## Nodes
* **`object_recognition_node`**

    Recognition at real-time.