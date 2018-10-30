#include "image2kinect/image2kinect.h"

Image2Kinect::Image2Kinect(ros::NodeHandle _nh) : node_handle(_nh), width(0), height(0)
{
    readParameters();

    object_recognition_sub = node_handle.subscribe(object_recognition_sub_topic, sub_queue_size, &Image2Kinect::objectRecognitionCallback, this);
    face_recognition_sub = node_handle.subscribe(face_recognition_sub_topic, sub_queue_size, &Image2Kinect::faceRecognitionCallback, this);
    people_tracking_sub = node_handle.subscribe(people_tracking_sub_topic, sub_queue_size, &Image2Kinect::peopleTrackingCallback, this);

    object_recognition_pub = node_handle.advertise<vision_system_msgs::Recognitions3D>(object_recognition_pub_topic, pub_queue_size);
    face_recognition_pub = node_handle.advertise<vision_system_msgs::Recognitions3D>(face_recognition_pub_topic, pub_queue_size);
    people_tracking_pub = node_handle.advertise<vision_system_msgs::Description3D>(people_tracking_pub_topic, pub_queue_size);

    image_request_client = node_handle.serviceClient<vision_system_msgs::ImageRequest>(image_request_client_service);
    segmentation_request_client = node_handle.serviceClient<vision_system_msgs::SegmentationRequest>(segmentation_request_client_service);

    camera_matrix_color = cv::Mat::zeros(3, 3, CV_64F);
}



void Image2Kinect::readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr& camera_info)
{
    bool recalculate_tabels = false;

    if(width != camera_info->width || height != camera_info->height){
        recalculate_tabels = true;
        width = camera_info->width;
        height = camera_info->height;
    } 

    double *it = camera_matrix_color.ptr<double>(0, 0);
    for(int i = 0 ; i < 9 ; i++, it++) {
        if(*it != camera_info->K[i]){
            recalculate_tabels = true;
            *it = camera_info->K[i];
        } 
    }

    if(recalculate_tabels) createTabels();
}

void Image2Kinect::createTabels()
{
    float fx = 1.0f / camera_matrix_color.at<double>(0, 0);
    float fy = 1.0f / camera_matrix_color.at<double>(1, 1);
    float cx = camera_matrix_color.at<double>(0, 2);
    float cy = camera_matrix_color.at<double>(1, 2);

    float* it;
    table_x = cv::Mat(1, width, CV_32F);
    it = table_x.ptr<float>();
    for(int c = 0 ; c < width ; c++, it++) {
        *it = (c - cx) * fx;
    }

    table_y = cv::Mat(1, height, CV_32F);
    it = table_y.ptr<float>();
    for(int r = 0 ; r < height ; r++, it++) {
        *it = (r - cy) * fy;
    }
}


void Image2Kinect::readImage(const sensor_msgs::Image::ConstPtr& msg_image, cv::Mat &image)
{
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(msg_image, msg_image->encoding);
    cv_image->image.copyTo(image);
}

void Image2Kinect::rgbd2Point(cv::Mat &color, cv::Mat &depth, geometry_msgs::Point &point)
{
    const float bad_point = std::numeric_limits<float>::quiet_NaN();

    geometry_msgs::Point &mean_position = point;

    mean_position.x = 0.0f;
    mean_position.y = 0.0f;
    mean_position.z = 0.0f;

    std::vector<geometry_msgs::Point> points;
    for(int r = 0 ; r < depth.rows ; r++) {

        uint16_t *it_depth = depth.ptr<uint16_t>(r);
        cv::Vec3b *it_color = color.ptr<cv::Vec3b>(r);

        for(int c = 0 ; c < depth.cols ; c++, it_depth++, it_color++) {
            if(it_color->val[0] != 0 || it_color->val[1] != 0 || it_color->val[2] != 0){
                geometry_msgs::Point point;
                
                if(*it_depth == 0) continue;

                float depth_value = *it_depth/1000.0;

                point.x = depth_value * table_x.at<float>(0, c);
                point.y = depth_value * table_y.at<float>(0, r);
                point.z = depth_value;

                mean_position.x += point.x;
                mean_position.y += point.y;
                mean_position.z += point.z;

                points.push_back(point);

            }
        }
    }

    if(points.size() <= 0) {
        mean_position.x = bad_point;
        mean_position.y = bad_point;
        mean_position.z = bad_point;
        return;
    } 

    mean_position.x /= points.size();
    mean_position.y /= points.size();
    mean_position.z /= points.size();

    /*std::vector<geometry_msgs::Point>::iterator it;

    for(it = points.begin() ; it != points.end() ; it++) {
        pose.covariance[6*0 + 0] += (it->x - mean_position.x)*(it->x - mean_position.x); //xx
        pose.covariance[6*0 + 1] += (it->x - mean_position.x)*(it->y - mean_position.y); //xy
        pose.covariance[6*0 + 2] += (it->x - mean_position.x)*(it->z - mean_position.z); //xz
        pose.covariance[6*1 + 1] += (it->y - mean_position.y)*(it->y - mean_position.y); //yy
        pose.covariance[6*1 + 2] += (it->y - mean_position.y)*(it->z - mean_position.z); //yz
        pose.covariance[6*2 + 2] += (it->z - mean_position.z)*(it->z - mean_position.z); //zz
    }

    pose.covariance[6*1 + 0] = pose.covariance[6*0 + 1]; //yx
    pose.covariance[6*2 + 0] = pose.covariance[6*0 + 2]; //zx
    pose.covariance[6*2 + 1] = pose.covariance[6*1 + 2]; //zy

    for(int i = 0 ; i < 3 ; i++) {
        for(int j = 0 ; j < 3 ; j++) {
            pose.covariance[6*i + j] /= points.size();
        }
    }*/

}


void Image2Kinect::recognitions2Recognitions3d(vision_system_msgs::Recognitions &recognitions, vision_system_msgs::Recognitions3D& recognitions3d)
{
    int frame_id = recognitions.image_header.seq;

    ROS_INFO("Frame ID: %d", frame_id);

    recognitions3d.header = recognitions.header;
    recognitions3d.image_header = recognitions.image_header;

    vision_system_msgs::ImageRequest image_srv;
    image_srv.request.seq = frame_id;
    if (!image_request_client.call(image_srv)) {
        ROS_ERROR("Failed to call image_request service");
        return;
    }

    cv::Mat segmented_rgb_image, depth, segmented_depth_image;

    vision_system_msgs::RGBDImage &rgbd_image = image_srv.response.rgbd_image;

    sensor_msgs::Image::ConstPtr depth_const_ptr( new sensor_msgs::Image(rgbd_image.depth));
    sensor_msgs::CameraInfo::ConstPtr camera_info_const_ptr( new sensor_msgs::CameraInfo(image_srv.response.camera_info));

    readImage(depth_const_ptr, depth);
    readCameraInfo(camera_info_const_ptr);

    std::vector<vision_system_msgs::Description> &descriptions = recognitions.descriptions;

    vision_system_msgs::SegmentationRequest segmentation_srv;
    segmentation_srv.request.initial_rgbd_image = rgbd_image;
    segmentation_srv.request.descriptions = descriptions;
    if (!segmentation_request_client.call(segmentation_srv)) {
        ROS_ERROR("Failed to call segmentation service");
        return;
    }

    std::vector<vision_system_msgs::Description>::iterator it;

    std::vector<sensor_msgs::Image> &segmented_rgb_images = segmentation_srv.response.segmented_rgb_images;
    std::vector<sensor_msgs::Image>::iterator jt;

  
    std::vector<vision_system_msgs::Description3D> &descriptions3d = recognitions3d.descriptions;

    std_msgs::Header point_header;
    point_header.seq = recognitions3d.header.seq;
    point_header.stamp = recognitions3d.image_header.stamp;
    point_header.frame_id = std::string("kinect_base");

    for(it = descriptions.begin(), jt = segmented_rgb_images.begin() ; it != descriptions.end() && jt != segmented_rgb_images.end() ; it++, jt++) {
        vision_system_msgs::Description3D description3d;
        description3d.label_class = it->label_class;
        description3d.probability = it->probability;
        geometry_msgs::PointStamped &point = description3d.position;
        point.header = point_header;

        cv::Rect roi;
        roi.x = (*it).bounding_box.minX;
        roi.y = (*it).bounding_box.minY;
        roi.width = (*it).bounding_box.width;
        roi.height = (*it).bounding_box.height;
        segmented_depth_image = depth(roi);
        sensor_msgs::Image::ConstPtr rgb_const_ptr( new sensor_msgs::Image(*jt));
        readImage(rgb_const_ptr, segmented_rgb_image);
	    cv::imshow("Seg", segmented_rgb_image);
	    cv::waitKey(1);

        rgbd2Point(segmented_rgb_image, segmented_depth_image, point.point);
        descriptions3d.push_back(description3d);
    }
}

void Image2Kinect::objectRecognitionCallback(vision_system_msgs::Recognitions recognitions)
{
    vision_system_msgs::Recognitions3D recognitions3d;
    segmentation_model_id = "median_center";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    object_recognition_pub.publish(recognitions3d);
}

void Image2Kinect::faceRecognitionCallback(vision_system_msgs::Recognitions recognitions)
{
    vision_system_msgs::Recognitions3D recognitions3d;
    segmentation_model_id = "median_center";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    face_recognition_pub.publish(recognitions3d);
}

void Image2Kinect::peopleTrackingCallback(vision_system_msgs::Recognitions recognitions)
{
    vision_system_msgs::Recognitions3D recognitions3d;
    segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    people_tracking_pub.publish(recognitions3d);
}

void Image2Kinect::readParameters()
{
    node_handle.param("/image2kinect/subscribers/queue_size", sub_queue_size, 5);
    node_handle.param("/image2kinect/subscribers/object_recognition/topic", object_recognition_sub_topic, std::string("/vision_system/or/object_recognition"));
    node_handle.param("/image2kinect/subscribers/face_recognition/topic", face_recognition_sub_topic, std::string("/vision_system/fr/face_recognition"));
    node_handle.param("/image2kinect/subscribers/people_tracking/topic", people_tracking_sub_topic, std::string("/vision_system/pt/people_tracking"));

    node_handle.param("/image2kinect/publishers/queue_size", pub_queue_size, 5);
    node_handle.param("/image2kinect/publishers/object_recognition/topic", object_recognition_pub_topic, std::string("/vision_system/or/object_recognition3d"));
    node_handle.param("/image2kinect/publishers/face_recognition/topic", face_recognition_pub_topic, std::string("/vision_system/fr/face_recognition3d"));
    node_handle.param("/image2kinect/publishers/people_tracking/topic", people_tracking_pub_topic, std::string("/vision_system/pt/people_tracking3d"));
    
    node_handle.param("/image2kinect/clients/image_request/service", image_request_client_service, std::string("/vision_system/is/image_request"));
    node_handle.param("/image2kinect/clients/segmentation_request/service", segmentation_request_client_service, std::string("/vision_system/seg/image_segmentation"));
}
