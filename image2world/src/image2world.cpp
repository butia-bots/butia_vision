#include "image2world/image2world.h"

Image2World::Image2World(ros::NodeHandle _nh) : node_handle(_nh), width(0), height(0)
{
    readParameters();
    camera_info_subscriber = node_handle.subscribe(camera_info_topic, camera_info_qs, &Image2World::cameraInfoCallback, this);
    image2world_server = node_handle.advertiseService(image2world_server_service, &Image2World::image2worldCallback, this);
    image_client = node_handle.serviceClient<vision_system_msgs::ImageRequest>(image_client_service);

    camera_matrix_color = cv::Mat::zeros(3, 3, CV_64F);
}

void Image2World::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info)
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

void Image2World::createTabels()
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


void Image2World::readImage(const sensor_msgs::Image::ConstPtr& msg_image, cv::Mat &image)
{
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(msg_image, msg_image->encoding);
    cv_image->image.copyTo(image);
}

void Image2World::rgbd2PoseWithCovariance(cv::Mat &color, cv::Mat &depth, geometry_msgs::PoseWithCovariance &pose)
{
    const float bad_point = std::numeric_limits<float>::quiet_NaN();

    geometry_msgs::Point &mean_position = pose.pose.position;

    mean_position.x = 0.0f;
    mean_position.y = 0.0f;
    mean_position.z = 0.0f;
    for(int i = 0 ; i < 36 ; i++){
        pose.covariance[i] = 0.0f;
    }

    std::vector<geometry_msgs::Point32> points;
    for(int r = 0 ; r < depth.rows ; r++) {

        uint16_t *it_depth = depth.ptr<uint16_t>(r);
        cv::Vec3b *it_color = color.ptr<cv::Vec3b>(r);

        for(int c = 0 ; c < depth.cols ; c++, it_depth++, it_color++) {
            if(it_color->val[0] != 0 || it_color->val[1] != 0 || it_color->val[2] != 0){
                geometry_msgs::Point32 point;
                
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

    std::vector<geometry_msgs::Point32>::iterator it;

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
    }
}

bool Image2World::image2worldCallback(vision_system_msgs::Image2World::Request &request, vision_system_msgs::Image2World::Response &response)
{
    int frame_id = request.recognitions.image_header.seq;

    ROS_INFO("Frame ID: %d", frame_id);

    vision_system_msgs::ImageRequest image_srv;
    image_srv.request.frame = frame_id;
    if (!image_client.call(image_srv)) {
        ROS_ERROR("Failed to call image_server service");
        return false;
    }

    cv::Mat color, depth, crop_color, crop_depth;

    vision_system_msgs::RGBDImage rgbd_image = image_srv.response.rgbd_image;

    sensor_msgs::Image::ConstPtr rgb_const_ptr( new sensor_msgs::Image(rgbd_image.rgb));
    sensor_msgs::Image::ConstPtr depth_const_ptr( new sensor_msgs::Image(rgbd_image.depth));

    readImage(rgb_const_ptr, color);
    readImage(depth_const_ptr, depth);

    //std::vector<sensor_msgs::PointCloud> clouds;
    std::vector<geometry_msgs::PoseWithCovariance> &poses = response.poses;

    std::vector<vision_system_msgs::Description> descriptions = request.recognitions.descriptions;
    std::vector<vision_system_msgs::Description>::iterator it;

    for(it = descriptions.begin() ; it != descriptions.end() ; it++) {
        //sensor_msgs::PointCloud cloud;
        geometry_msgs::PoseWithCovariance pose;
        cv::Rect roi;
        roi.x = (*it).bounding_box.minX;
        roi.y = (*it).bounding_box.minY;
        roi.width = (*it).bounding_box.width;
        roi.height = (*it).bounding_box.height;
        crop_color = color(roi);
        crop_depth = depth(roi);

        //segment(crop_color)

        rgbd2PoseWithCovariance(crop_color, crop_depth, pose);
        //botar campos que faltam na cloud
        poses.push_back(pose);
    }

    return true;
}

void Image2World::readParameters()
{
    node_handle.param("/object_recognition/subscribers/camera_info/topic", camera_info_topic, std::string("/kinect2/qhd/camera_info"));
    node_handle.param("/object_recognition/subscribers/camera_info/qs", camera_info_qs, 1);

    node_handle.param("/object_recognition/services/image2world/service", image2world_server_service, std::string("/vision_system/iw/image2world"));

    node_handle.param("/object_recognition/services/image_server/service", image_client_service, std::string("/vision_system/is/image_request"));

    node_handle.param("/object_recognition/services/segmentation_server/service", segmentation_client_service, std::string(""));
}
