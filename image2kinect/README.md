# image2kinect
## Overview
The image2kinect is used to transform recognitions in the image plan to recognitions in the world with kinect as the origin.
## Topics
### Subscribed Topics
* **`object_recognition:`** `/vision_system/or/object_recognition` ([vision_system_msgs::Recognitions])

    Reads the recognized objects.

* **`face_recognition:`** `/vision_system/fr/face_recognition` ([vision_system_msgs::Recognitions])

    Reads the recognized faces.
    
* **`people_tracking:`** `/vision_system/pt/people_tracking` ([vision_system_msgs::Recognitions])

    Reads the recognition of tracked person.

### Published Topics
* **`object_recognition:`** `/vision_system/or/object_recognition3d` ([vision_system_msgs::Recognitions3D])

    Publishes the recognized objects in 3D.

* **`face_recognition:`** `/vision_system/fr/face_recognition3d` ([vision_system_msgs::Recognitions3D])

    Publishes the recognized faces in 3D.
    
* **`people_tracking:`** `/vision_system/pt/people_tracking3d` ([vision_system_msgs::Recognitions3D])

    Publishes the recognition of tracked person in 3D.

### Clients
* **`image_request:`** `/vision_system/vsb/image_request` ([vision_system_msgs::ImageRequest])

    Requests a RGBD image and a camera info of the readed seq in Recognitions.

## Nodes
* **`image2kinect_node`**

    Image to Kinect in real_time of the recognitions.