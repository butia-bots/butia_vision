# people_tracking
## Overview
The people_tracking package uses OpenCV and object_recognition package to get the bounding boxes of the people in the frame and extract, match and register the SIFT features of the people.

## Topics
### Subscribed Topics
* **`people_detection:`** `/vision_system/or/people_detection` ([vision_system_msgs::Recognitions])

    Publishes the recognized people.
### Published Topics
* **`people_tracking:`** `/vision_system/pt/people_tracking` ([vision_system_msgs::Recognitions])

    Publishes the matched person.

## Servers
* **`start_tracking:`** `/vision_system/pt/start` ([vision_system_msgs::StartTracking])

    Ask for the start of the people tracking.

* **`start_tracking:`** `/vision_system/pt/stop` ([vision_system_msgs::StopTracking])

    Ask for the stop of the people tracking.

## Nodes
* **`people_tracking_node`**

    Tracking at real-time.