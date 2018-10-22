# people_tracking
## Overview
The people_tracking package uses OpenCV and object_recognition package to get the bounding boxes of the people in the frame and extract, match and register the SIFT features of the people.
## Topics
### Subscribed Topics
* **`people_detection:`** `/vision_system/or/people_detection` ([vision_system_msgs::Recognitions])

    Publishes the recognized people.
### Published Topics
* **`people_tracking:`** `/vision_system/pt/person_pose` ([vision_system_msgs::PersonPose])

    Publishes the position of a person in the world.
### Nodes
* **`people_tracking_node`**

    Tracking at real-time.