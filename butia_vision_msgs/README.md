# butia_vision_msgs
## Overview
The butia_vision_msgs package has all messages and services used by the system.
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

### RGBDImage
    sensor_msgs/Image rgb
    sensor_msgs/Image depth

## Services
### FaceClassifierTraining
    string LINEAR_SVM = 'lsvm'
    string RADIAL_SVM = 'rsvm'
    string GRID_SEARCH_SVM = 'gssvm'
    string GAUSSIAN_MIXTURE_MODELS = 'gmm'
    string DECISION_TREE = 'dt'
    string GAUSSIAN_NAIVE_BAYES = 'gnb'
    string K_NEAREST_NEIGHBORS = 'knn'
    string classifier_type
    string classifier_name
    ---
    bool sucess

### ImageRequest
    uint64 seq
    ---
    RGBDImage rgbd_image
    sensor_msgs/CameraInfo camera_info

### PeopleIntroducing
    string LINEAR_SVM = 'lsvm'
    string RADIAL_SVM = 'rsvm'
    string GRID_SEARCH_SVM = 'gssvm'
    string GAUSSIAN_MIXTURE_MODELS = 'gmm'
    string DECISION_TREE = 'dt'
    string GAUSSIAN_NAIVE_BAYES = 'gnb'
    string K_NEAREST_NEIGHBORS = 'knn'
    string name
    int32 num_images
    string classifier_type
    ---
    bool response

### SegmentationRequest
    RGBDImage initial_rgbd_image
    Description[] descriptions
    ---
    sensor_msgs/Image[] segmented_rgb_images

### StartTracking
    bool start
    ---
    bool started

### StopTracking
    bool stop
    ---
    bool stopped
