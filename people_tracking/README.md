# people_tracking
## Overview
The people_tracking package uses [OpenCV](https://github.com/opencv/opencv), object_recognition package and [deep SORT algorithm](https://github.com/nwojke/deep_sort) to get the bounding boxes of the people in the frame, apply tracking algorithm and extract features of the people in image.

## Topics
### Subscribed Topics
* **`people_detection:`** `/butia_vision/or/people_detection` ([butia_vision_msgs::Recognitions])

    Publishes the recognized people.
### Published Topics
* **`people_tracking:`** `/butia_vision/pt/people_tracking` ([butia_vision_msgs::Recognitions])

    Publishes the matched person.

## Servers
* **`start_tracking:`** `/butia_vision/pt/start` ([butia_vision_msgs::StartTracking])

    Ask for the start of the people tracking.

* **`start_tracking:`** `/butia_vision/pt/stop` ([butia_vision_msgs::StopTracking])

    Ask for the stop of the people tracking.

## Nodes
* **`people_tracking_node`**

    Tracking at real-time.
