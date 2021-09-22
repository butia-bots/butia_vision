#include "image2kinect/image2kinect.h"

Image2Kinect::Image2Kinect(ros::NodeHandle _nh) : node_handle(_nh), width(0), height(0)
{
    readParameters();

    object_recognition_sub = node_handle.subscribe(object_recognition_sub_topic, sub_queue_size, &Image2Kinect::objectRecognitionCallback, this);
    face_recognition_sub = node_handle.subscribe(face_recognition_sub_topic, sub_queue_size, &Image2Kinect::faceRecognitionCallback, this);
    people_tracking_sub = node_handle.subscribe(people_tracking_sub_topic, sub_queue_size, &Image2Kinect::peopleTrackingCallback, this);

    object_recognition_pub = node_handle.advertise<butia_vision_msgs::Recognitions3D>(object_recognition_pub_topic, pub_queue_size);
    face_recognition_pub = node_handle.advertise<butia_vision_msgs::Recognitions3D>(face_recognition_pub_topic, pub_queue_size);
    people_tracking_pub = node_handle.advertise<butia_vision_msgs::Recognitions3D>(people_tracking_pub_topic, pub_queue_size);

    image_request_client = node_handle.serviceClient<butia_vision_msgs::ImageRequest>(image_request_client_service);
    //segmentation_request_client = node_handle.serviceClient<butia_vision_msgs::SegmentationRequest>(segmentation_request_client_service);
}

bool Image2Kinect::points2RGBPoseWithCovariance(PointCloud &points, butia_vision_msgs::BoundingBox &bb, geometry_msgs::PoseWithCovariance &pose, std_msgs::ColorRGBA &color, cv::Mat &mask)
{
    geometry_msgs::Point &mean_position = pose.pose.position;
    geometry_msgs::Quaternion &orientation = pose.pose.orientation;
    std_msgs::ColorRGBA &mean_color = color;

    mean_position.x = 0.0f;
    mean_position.y = 0.0f;
    mean_position.z = 0.0f;

    mean_color.r = 0.0f;
    mean_color.g = 0.0f;
    mean_color.b = 0.0f;

    bool use_mask = false;
    if(mask.rows*mask.cols != 0) use_mask = true;

    int min_x, min_y, max_x, max_y;

    if(!use_mask) {
        min_x = (bb.minX + bb.width/2 - kernel_size/2) < bb.minX ? bb.minX : (bb.minX + bb.width/2 - kernel_size/2);
        min_y = (bb.minY + bb.height/2 - kernel_size/2) < bb.minY ? bb.minY : (bb.minY + bb.height/2 - kernel_size/2);
        max_x = (bb.minX + bb.width/2 + kernel_size/2) > (bb.minX + bb.width) ? (bb.minX + bb.width) : (bb.minX + bb.width/2 + kernel_size/2);
        max_y = (bb.minY + bb.height/2 + kernel_size/2) > (bb.minY + bb.height) ? (bb.minY + bb.height) : (bb.minY + bb.height/2 + kernel_size/2);

        min_x = min_x < 0 ? 0 : min_x;
        min_y = min_y < 0 ? 0 : min_y;

        max_x = max_x >= points.width ? points.width - 1 : max_x;
        max_y = max_y >= points.height ? points.height - 1 : max_y;
    }
    else {
        min_x = min_x;
        min_y = min_y;

        max_x = min_x + points.width - 1;
        max_y = min_y + points.height - 1; 
    }
    

    std::vector<pcl::PointXYZRGB> valid_points;
    for(int r = min_y; r <= max_y ; r++) {
        for(int c = min_x; c <= max_x ; c++) {
            pcl::PointXYZRGB point = points.at(c, r);
            float depth = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            if((use_mask && mask.at<int>(r - min_y,c - min_x) != 0) || !use_mask) {
                if(depth <= max_depth && depth != 0){     
                    mean_position.x += point.x;
                    mean_position.y += point.y;
                    mean_position.z += point.z;

                    mean_color.r += point.r;
                    mean_color.g += point.g;
                    mean_color.b += point.b;
                    valid_points.push_back(point);
                }
            }
        }
    }

    if(valid_points.size() == 0) {
        std::cout<< "No valid point." << std::endl;
        return false;
    } 

    mean_position.x /= valid_points.size();
    mean_position.y /= valid_points.size();
    mean_position.z /= valid_points.size();

    orientation.x = 0;
    orientation.y = 0;
    orientation.z = 0;
    orientation.w = 1;

    mean_color.r /= valid_points.size();
    mean_color.g /= valid_points.size();
    mean_color.b /= valid_points.size();

    //review this, may be wrong
    std::vector<pcl::PointXYZRGB>::iterator it;
    for(it = valid_points.begin() ; it != valid_points.end() ; it++) {
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
            pose.covariance[6*i + j] /= valid_points.size();
        }
    }

    return true;
}


void Image2Kinect::readImage(const sensor_msgs::Image::ConstPtr& msg_image, cv::Mat &image)
{
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(msg_image, msg_image->encoding);
    cv_image->image.copyTo(image);
}

void Image2Kinect::readPoints(const sensor_msgs::PointCloud2::ConstPtr& msg_points, PointCloud &points)
{
    pcl::PCLPointCloud2 cloud; 
    pcl_conversions::toPCL(*msg_points, cloud);
    pcl::fromPCLPointCloud2(cloud, points);
}


void Image2Kinect::recognitions2Recognitions3d(butia_vision_msgs::Recognitions &recognitions, butia_vision_msgs::Recognitions3D& recognitions3d)
{
    int frame_id = recognitions.image_header.seq;

    ROS_INFO("Frame ID: %d", frame_id);

    recognitions3d.header = recognitions.header;
    recognitions3d.image_header = recognitions.image_header;

    butia_vision_msgs::ImageRequest image_srv;
    image_srv.request.seq = frame_id;
    if (!image_request_client.call(image_srv)) {
        ROS_ERROR("FAILED TO CALL IMAGE REQUEST SERVICE");
        return;
    }

    cv::Mat segmented_rgb_image, depth, segmented_depth_image;
    PointCloud points;

    butia_vision_msgs::RGBDImage &rgbd_image = image_srv.response.rgbd_image;
    sensor_msgs::Image::ConstPtr depth_const_ptr( new sensor_msgs::Image(rgbd_image.depth));
    sensor_msgs::PointCloud2::ConstPtr points_const_ptr( new sensor_msgs::PointCloud2(image_srv.response.points));

    readImage(depth_const_ptr, depth);
    readPoints(points_const_ptr, points);

    std::vector<butia_vision_msgs::Description> &descriptions = recognitions.descriptions;

    // butia_vision_msgs::SegmentationRequest segmentation_srv;
    // segmentation_srv.request.initial_rgbd_image = rgbd_image;
    // segmentation_srv.request.descriptions = descriptions;
    // if (!segmentation_request_client.call(segmentation_srv)) {
    //     ROS_ERROR("Failed to call segmentation service");
    //     return;
    // }

    //here, it is possible to apply point cloud filters, as: cut-off, voxel grid, statistical and others

    std::vector<butia_vision_msgs::Description>::iterator it;

    // std::vector<sensor_msgs::Image> &segmented_rgb_images = segmentation_srv.response.segmented_rgb_images;
    // std::vector<sensor_msgs::Image>::iterator jt;
  
    std::vector<butia_vision_msgs::Description3D> &descriptions3d = recognitions3d.descriptions;


    for(it = descriptions.begin(); it != descriptions.end(); it++) {
        butia_vision_msgs::Description3D description3d;
        description3d.label_class = it->label_class;
        description3d.probability = it->probability;
        std_msgs::ColorRGBA &color =  description3d.color;
        geometry_msgs::PoseWithCovariance &pose = description3d.pose;
        cv::Mat mask;
        sensor_msgs::Image::ConstPtr mask_const_ptr( new sensor_msgs::Image(it->mask));
        readImage(mask_const_ptr, mask);

        // cv::Rect roi;
        // roi.x = (*it).bounding_box.minX;
        // roi.y = (*it).bounding_box.minY;
        // roi.width = (*it).bounding_box.width;
        // roi.height = (*it).bounding_box.height;
        // segmented_depth_image = depth(roi);

        // sensor_msgs::Image::ConstPtr rgb_const_ptr( new sensor_msgs::Image(*jt));
        // readImage(rgb_const_ptr, segmented_rgb_image);

        if(points2RGBPoseWithCovariance(points, (*it).bounding_box, pose, color, mask))
            descriptions3d.push_back(description3d);
    }

    if(publish_tf)
        publishTF(recognitions3d);
}

void Image2Kinect::publishTF(butia_vision_msgs::Recognitions3D &recognitions3d)
{
    std::vector<butia_vision_msgs::Description3D> &descriptions3d = recognitions3d.descriptions;
    std::vector<butia_vision_msgs::Description3D>::iterator it;

    std::map<std::string, int> current_rec;

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;

    for(it = descriptions3d.begin() ; it != descriptions3d.end() ; it++) {
        if(current_rec.find(it->label_class) == current_rec.end()) {
            current_rec[it->label_class] = 0;
        }
        else {
            current_rec[it->label_class]++;
        }

        transform.setOrigin( tf::Vector3(it->pose.pose.position.x, it->pose.pose.position.y, it->pose.pose.position.z) );
        q.setRPY(0, 0, 0);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), recognitions3d.image_header.frame_id,
                                              it->label_class + std::to_string(current_rec[it->label_class]) + "_detect"));
    }
}

void Image2Kinect::objectRecognitionCallback(butia_vision_msgs::Recognitions recognitions)
{
    butia_vision_msgs::Recognitions3D recognitions3d;
    //segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    object_recognition_pub.publish(recognitions3d);
}

void Image2Kinect::faceRecognitionCallback(butia_vision_msgs::Recognitions recognitions)
{
    butia_vision_msgs::Recognitions3D recognitions3d;
    //segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    face_recognition_pub.publish(recognitions3d);
}

void Image2Kinect::peopleTrackingCallback(butia_vision_msgs::Recognitions recognitions)
{
    butia_vision_msgs::Recognitions3D recognitions3d;
    //segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    people_tracking_pub.publish(recognitions3d);
}

void Image2Kinect::readParameters()
{
    node_handle.param("/image2kinect/subscribers/queue_size", sub_queue_size, 5);
    node_handle.param("/image2kinect/subscribers/object_recognition/topic", object_recognition_sub_topic, std::string("/butia_vision/or/object_recognition"));
    node_handle.param("/image2kinect/subscribers/face_recognition/topic", face_recognition_sub_topic, std::string("/butia_vision/fr/face_recognition"));
    node_handle.param("/image2kinect/subscribers/people_tracking/topic", people_tracking_sub_topic, std::string("/butia_vision/pt/people_tracking"));

    node_handle.param("/image2kinect/publishers/queue_size", pub_queue_size, 5);
    node_handle.param("/image2kinect/publishers/object_recognition/topic", object_recognition_pub_topic, std::string("/butia_vision/or/object_recognition3d"));
    node_handle.param("/image2kinect/publishers/face_recognition/topic", face_recognition_pub_topic, std::string("/butia_vision/fr/face_recognition3d"));
    node_handle.param("/image2kinect/publishers/people_tracking/topic", people_tracking_pub_topic, std::string("/butia_vision/pt/people_tracking3d"));
    
    node_handle.param("/image2kinect/clients/image_request/service", image_request_client_service, std::string("/butia_vision/is/image_request"));
    //node_handle.param("/image2kinect/clients/segmentation_request/service", segmentation_request_client_service, std::string("/butia_vision/seg/image_segmentation"));

    //node_handle.param("/image2kinect/segmentation_threshold", segmentation_threshold, (float)0.2);
    node_handle.param("/image2kinect/max_depth", max_depth, 2000);
    node_handle.param("/image2kinect/publish_tf", publish_tf, true);
    node_handle.param("/image2kinect/kernel_size", kernel_size, 5);

}
