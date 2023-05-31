# shelf_detection
## Overview
The shelf_detection package uses OpenCV and object_recognition package to detect levels in the shelf and its object type by the objects detected in the level.

## Topics
### Subscribed Topics
* **`object_recognition:`** `/butia_vision/or/object_recognition` ([butia_vision_msgs::Recognitions])

    Publishes the recognized objects.
### Published Topics
* **`shelf_detection:`** `/butia_vision/sd/shelf_detection` ([butia_vision_msgs::Shelf])

    Publishes the shelf features.

## Nodes
* **`shekf_detection_node`**

    Detection at real-time.