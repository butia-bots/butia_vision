# vision_system_msgs
## Overview
The vision_system_msgs package has all messages and services used by the system.
## Messages
### BoundingBox
    int32 minX
    int32 minY
    int32 width
    int32 height

### ClassifierReload
    string model_name

### Description
    string label_class
    float64 probability
    BoundingBox bounding_box

### Description3D
    string label_class
    float64 probability
    geometry_msgs/PoseWithCovariance pose

### Recognitions
    Header image_header
    Header recognition_header
    Description[] descriptions

### Recognitions3D
    Header image_header
    Header recognition_header
    Description3D[] descriptions

## Services
### FaceClassifierTraining
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
### Image2World
    Recognitions recognitions
    ---
    geometry_msgs/PoseWithCovariance[] poses
### ImageRequest
    uint64 frame
    ---
    RGBDImage rgbd_image