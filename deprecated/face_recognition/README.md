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

* **`classifier_reload:`** `/butia_vision/fr/classifier_reload` ([butia_vision_msgs::ClassifierReload])

    Ask for change of classifier model.

## Published Topics
* **`face_recognition:`** `/butia_vision/fr/face_recognition` ([butia_vision_msgs::Recognitions])

    Publishes the recognized faces.

* **`face_recognition_view:`** `/butia_vision/fr/face_recognition_view` ([sensor_msgs::Image])

    Publishes an image of the recognized faces to debug the process.

* **`classifier_reload:`** `/butia_vision/fr/classifier_reload` ([butia_vision_msgs::ClassifierReload])

    Ask for change of classifier model.

* **`face_list_updated:`** `/butia_vision/fr/face_list_updated` ([std_msgs::Header])

    Version control of the face list.
    
## Servers
* **`people_introducing:`** `/butia_vision/fr/people_introducing` ([butia_vision_msgs::PeopleIntroducing])

    Responds for the system meet a new person and training of a new classifier model after.

* **`classifier_training:`** `/butia_vision/fr/classifier_training` ([butia_vision_msgs::FaceClassifierTraining])

    Responds for training of a new classifier model.

* **`list_faces:`** `/butia_vision/fr/list_faces` ([butia_vision_msgs::ListClasses])

    Responds with the list of possible faces to recognize.

## Nodes
* **`face_recognition_node`**

    Recognition at real-time.
* **`classifier_training_node`**

    Allow to train a new model of face classification.