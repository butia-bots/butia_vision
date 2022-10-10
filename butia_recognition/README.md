# object_recognition
## Overview
The object_recognition package uses darknet_ros information and redistribute in two topics, one for objects and other for people.

## Topics
### Subscribed Topics
* **`bounding_boxes:`** `/darknet_ros/bounding_boxes` ([darknet_ros_msgs::BoundingBoxes])

    Reads an array of bounding boxes that gives information of the position, class and size of the object in pixel coordinates.

### Published Topics
* **`object_recognition:`** `/butia_vision/or/object_recognition` ([butia_vision_msgs::Recognitions])

    Publishes the recognized objects.
    
* **`people_detection:`** `/butia_vision/or/people_detection` ([butia_vision_msgs::Recognitions])

    Publishes the recognized people.

* **`object_list_updated:`** `/butia_vision/or/object_list_updated` ([std_msgs::Header])

    Version control of the object list.

## Servers
* **`list_objects:`** `/butia_vision/or/list_objects` ([butia_vision_msgs::ListClasses])

    Responds with the list of possible objects to recognize.

## Nodes
* **`object_recognition_node`**

    Recognition at real-time.