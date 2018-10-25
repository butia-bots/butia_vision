# object_recognition
## Overview
The object_recognition package uses darknet_ros information and redistribute in two topics, one for objects and other for people.
## Topics
### Subscribed Topics
* **`bounding_boxes:`** `/darknet_ros/bounding_boxes` ([darknet_ros_msgs::BoundingBoxes])

    Reads an array of bounding boxes that gives information of the position, class and size of the object in pixel coordinates.

### Published Topics
* **`object_recognition:`** `/vision_system/or/object_recognition` ([vision_system_msgs::Recognitions])

    Publishes the recognized objects.
    
* **`people_detection:`** `/vision_system/or/people_detection` ([vision_system_msgs::Recognitions])

    Publishes the recognized people.

## Nodes
* **`object_recognition_node`**

    Recognition at real-time.