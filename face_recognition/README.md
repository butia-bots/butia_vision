# face_recognition
## Overview
The face_recognition package uses Openface, OpenCV and scikit-learn to perform all four parts in a face recognize process:
- detection (openface or opencv)
- alignment (openface)
- embeding (openface)
- classification (scikit-learn)

## Topics
### Subscribed Topics
* **`camera_reading:`** `/camera/rgb/image_raw` ([sensor_msgs/Image])

    The camera measurements.

* **`classifier_reload:`** `/vision_system/fr/classifier_reload` ([vision_system_msgs/ClassifierReload])

    Ask for change of classifier model.

## Published Topics
* **`face_recognition:`** `/vision_system/fr/face_recognition` ([vision_system_msgs::Recognitions])

    Publishes the recognized faces.

* **`face_recognition_view:`** `/vision_system/fr/face_recognition_view` ([sensor_msgs::Image])

    Publishes an image of the recognized faces to debug the process.

* **`classifier_reload:`** `/vision_system/fr/classifier_reload` ([vision_system_msgs/ClassifierReload])

    Ask for change of classifier model.
    
## Services
* **`classifier_training:`** `/vision_system/fr/classifier_training` ([vision_system_msgs::ClassifierTraining])


    Ask for training of a new classifier model.
## Nodes
* **`face_recognition_node`**

    Recognition at real-time.
* **`classifier_training_node`**

    Allow to train a new model of face classification.