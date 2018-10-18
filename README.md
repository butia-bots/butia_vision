# vision_system

## Overview
This is a group of ROS packages responsable for perform computer vision process of Butia Bots domestic robot (DoRIS) in Robocup@Home league. 

**Author: [Igor Maurell], igormaurell@furg.br**
**Author: [Miguel Martins], migueldossantos@furg.br**

## Dependencies
This software is built on the Robotic Operating System ([ROS]), which needs to be [installed](http://wiki.ros.org) first. Additionally, the packages depends of a few libraries and frameworks:

- [OpenCV](http://opencv.org/) (computer vision library)
- [Openface](https://cmusatyalab.github.io/openface/) (face recognition library)
- [scikit-learn](http://scikit-learn.org/stable/) (machine learning library)
- [darknet_ros](https://github.com/leggedrobotics/darknet_ros) (darknet ros package)

## Packages
This is a brief description of the packages. 

### vision_system_msgs
#### Overview
The vision_system_msgs package has all messages and services used by the system.
#### Messages
##### BoundingBox
    int32 minX
    int32 minY
    int32 width
    int32 height

##### ClassifierReload
    string model_name

##### Description
    string label_class
    float64 probability
    BoundingBox bounding_box

##### Description3D
    string label_class
    float64 probability
    geometry_msgs/PoseWithCovariance pose

##### Recognitions
    Header image_header
    Header recognition_header
    Description[] descriptions

##### Recognitions3D
    Header image_header
    Header recognition_header
    Description3D[] descriptions

#### Services
##### FaceClassifierTraining
    string LINEAR_SVM = 'lsvm'
    string RADIAL_SVM = 'rsvm'
    string GRID_SEARCH_SVM = 'gssvm'
    string GAUSSIAN_MIXTURE_MODELS = 'gmm'
    string DECISION_TREE = 'dt'
    string GAUSSIAN_NAIVE_BAYES = 'gnb'
    string classifier_type
    string classifier_name
    ---
    bool sucess

### face_recognition
#### Overview
The face_recognition package uses Openface, OpenCV and scikit-learn to perform all four parts in a face recognize process:
- detection (openface or opencv)
- alignment (openface)
- embeding (openface)
- classification (scikit-learn)

#### Topics
##### Subscribed Topics
* **`camera_reading:`** `/camera/rgb/image_raw` ([sensor_msgs/Image])

    The camera measurements.

* **`classifier_reload:`** `/vision_system/fr/classifier_reload` ([vision_system_msgs/ClassifierReload])

    Ask for change of classifier model.

#### Published Topics
* **`face_recognition:`** `/vision_system/fr/face_recognition` ([vision_system_msgs::Recognitions])

    Publishes the recognized faces.

* **`face_recognition3d:`** `/vision_system/fr/face_recognition3d` ([vision_system_msgs::Recognitions3D])

    Publishes the recognized faces in 3D.

* **`face_recognition_view:`** `/vision_system/fr/face_recognition_view` ([sensor_msgs::Image])

    Publishes an image of the recognized faces to debug the process.

* **`classifier_reload:`** `/vision_system/fr/classifier_reload` ([vision_system_msgs/ClassifierReload])

    Ask for change of classifier model.
    
#### Services
* **`classifier_training:`** `/vision_system/fr/classifier_training` ([vision_system_msgs::ClassifierTraining])


    Ask for training of a new classifier model.
#### Nodes
* **`face_recognition_node`**

    Recognition at real-time.
* **`classifier_training_node`**

    Allow to train a new model of face classification.
    
### object_recognition
#### Overview
The object_recognition package uses darknet_ros information and redistribute in two topics, one for objects and other for people.
#### Topics
##### Subscribed Topics
* **`bounding_boxes:`** `/darknet_ros/bounding_boxes` ([darknet_ros_msgs::BoundingBoxes])

    Publishes an array of bounding boxes that gives information of the position and size of the bounding box in pixel coordinates.
##### Published Topics
* **`object_recognition:`** `/vision_system/or/object_recognition` ([vision_system_msgs::Recognitions])

    Publishes the recognized objects.

* **`object_recognition3d:`** `/vision_system/or/object_recognition3d` ([vision_system_msgs::Recognitions3D])

    Publishes the recognized objects in 3d.
    
* **`people_detection:`** `/vision_system/or/people_detection` ([vision_system_msgs::Recognitions])

    Publishes the recognized people.
#### Nodes
* **`object_recognition_node`**

    Recognition at real-time.
### people_tracking
#### Overview
The people_tracking package uses OpenCV and object_recognition package to get the bounding boxes of the people in the frame and extract, match and register the SIFT features of the people.
#### Topics
##### Subscribed Topics
* **`people_detection:`** `/vision_system/or/people_detection` ([vision_system_msgs::Recognitions])

    Publishes the recognized people.
##### Published Topics
* **`people_tracking:`** `/vision_system/pt/person_pose` ([vision_system_msgs::PersonPose])

    Publishes the position of a person in the world.
#### Nodes
* **`people_tracking_node`**

    Tracking at real-time.
    

### img_server
#### Overview
The image_server package is an internal image server to support the system to get the images in real time.
#### Services
* **`image_request:`** `/vision_system/img_server/image_request` ([vision_system_msgs::ImageRequest])

    Ask for a frame from the image server.
#### Nodes
* **`img_server_node`**

    Updates the internal image server at real time.
