# face_recognition
## Overview
The face_recognition package uses Openface, dlib, OpenCV and scikit-learn to perform all four parts in a face recognize process:
- detection (openface or opencv)
- alignment (openface)
- embeding (openface)
- classification (scikit-learn)

## Topics
### Subscribed Topics
* **`camera_reading:`** `/camera/rgb/image_raw` ([sensor_msgs::Image])

    The camera measurements.

* **`classifier_reload:`** `/vision_system/fr/classifier_reload` ([vision_system_msgs::ClassifierReload])

    Ask for change of classifier model.

* **`face_list_updated:`** `/vision_system/fr/face_list_updated` ([std_msgs::Header])

    Version control of the face list.

## Published Topics
* **`face_recognition:`** `/vision_system/fr/face_recognition` ([vision_system_msgs::Recognitions])

    Publishes the recognized faces.

* **`face_recognition_view:`** `/vision_system/fr/face_recognition_view` ([sensor_msgs::Image])

    Publishes an image of the recognized faces to debug the process.

* **`classifier_reload:`** `/vision_system/fr/classifier_reload` ([vision_system_msgs::ClassifierReload])

    Ask for change of classifier model.
    
## Servers
* **`people_introducing:`** `/vision_system/fr/people_introducing` ([vision_system_msgs::PeopleIntroducing])

    Responds for the system meet a new person and training of a new classifier model after.

* **`classifier_training:`** `/vision_system/fr/classifier_training` ([vision_system_msgs::FaceClassifierTraining])

    Responds for training of a new classifier model.

* **`list_faces:`** `/vision_system/fr/list_faces` ([vision_system_msgs::ListClasses])

    Responds with the list of possible faces to recognize.

## Nodes
* **`face_recognition_node`**

    Recognition at real-time.
* **`classifier_training_node`**

    Allow to train a new model of face classification.