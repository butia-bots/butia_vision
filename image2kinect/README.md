# image2kinect
## Overview
The image2kinect is used to transform recognitions in the image plan to recognitions in the world with kinect as the origin.
## Topics
### Subscribed Topics
* **`object_recognition:`** `/butia_vision/or/object_recognition` ([butia_vision_msgs::Recognitions])

    Reads the recognized objects.

* **`face_recognition:`** `/butia_vision/fr/face_recognition` ([butia_vision_msgs::Recognitions])

    Reads the recognized faces.
    
* **`people_tracking:`** `/butia_vision/pt/people_tracking` ([butia_vision_msgs::Recognitions])

    Reads the recognition of tracked person.

### Published Topics
* **`object_recognition:`** `/butia_vision/or/object_recognition3d` ([butia_vision_msgs::Recognitions3D])

    Publishes the recognized objects in 3D.

* **`face_recognition:`** `/butia_vision/fr/face_recognition3d` ([butia_vision_msgs::Recognitions3D])

    Publishes the recognized faces in 3D.
    
* **`people_tracking:`** `/butia_vision/pt/people_tracking3d` ([butia_vision_msgs::Recognitions3D])

    Publishes the recognition of tracked person in 3D.

### Clients
* **`image_request:`** `/butia_vision/vsb/image_request` ([butia_vision_msgs::ImageRequest])

    Requests a RGBD image and a camera info of the readed seq in Recognitions.

## Nodes
* **`image2kinect_node`**

    Image to Kinect in real_time of the recognitions.