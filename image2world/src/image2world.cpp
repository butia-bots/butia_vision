#include "image2world/image2world.h"

Image2World::Image2World(ros::NodeHandle _nh) : node_handle(_nh), width(0), height(0)
{
    readParameters();
    /*
    image_d = node_handle.subscribe("/kinect2/qhd/image_color_rect", 1, &Image2World::imageRCb, this);
    image_r = node_handle.subscribe("/kinect2/qhd/image_depth_rect", 1, &Image2World::imageDCb, this);
    */
    camera_info_subscriber = node_handle.subscribe(camera_info_topic, camera_info_qs, &Image2World::cameraInfoCallback, this);
    image2world_server = node_handle.advertiseService(image2world_server_service, &Image2World::image2worldCallback, this);
    image_client = node_handle.serviceClient<vision_system_msgs::ImageRequest>(image_client_service);

    camera_matrix_color = cv::Mat::zeros(3, 3, CV_64F);
}

/*
void Image2World::imageDCb(const sensor_msgs::Image::ConstPtr &msg_image)
{
    ROS_INFO("D ID: %d", msg_image->header.seq);
    cv::Mat image;
    readImage(msg_image, image);
    cv::imshow("Depth", image);
    cv::waitKey(10);

}
void Image2World::imageRCb(const sensor_msgs::Image::ConstPtr &msg_image)
{
    ROS_INFO("RGB ID: %d", msg_image->header.seq);
    cv::Mat image;
    readImage(msg_image, image);
    cv::imshow("RGB", image);
    cv::waitKey(10);
}*/

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

void Image2World::rgb2PointCloud(cv::Mat &color, cv::Mat &depth, sensor_msgs::PointCloud& point_cloud)
{
    std::vector<geometry_msgs::Point32> &points = point_cloud.points;
    for(int r = 0 ; r < depth.rows ; r++) {

        uint16_t *it_depth = depth.ptr<uint16_t>(r);
        cv::Vec3b *it_color = color.ptr<cv::Vec3b>(r);

        for(int c = 0 ; c < depth.cols ; c++, it_depth++, it_color++) {
            if(it_color->val[0] != 0 || it_color->val[1] != 0 || it_color->val[2] != 0){
                geometry_msgs::Point32 point;
                float depth_value = *it_depth/1000.0;

                point.x = depth_value * table_x.at<float>(0, c);
                point.y = depth_value * table_y.at<float>(0, r);
                point.z = depth_value;

                points.push_back(point);

            }
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

    std::vector<sensor_msgs::PointCloud> clouds;

    std::vector<vision_system_msgs::Description> descriptions = request.recognitions.descriptions;
    std::vector<vision_system_msgs::Description>::iterator it;

    for(it = descriptions.begin() ; it != descriptions.end() ; it++) {
        sensor_msgs::PointCloud cloud;
        cv::Rect roi;
        roi.x = (*it).bounding_box.minX;
        roi.y = (*it).bounding_box.minY;
        roi.width = (*it).bounding_box.width;
        roi.height = (*it).bounding_box.height;
        crop_color = color(roi);
        crop_depth = depth(roi);

        //segment(crop_color)

        rgb2PointCloud(crop_color, crop_depth, cloud);
        //botar campos que faltam na cloud
        clouds.push_back(cloud);

    }

    response.clouds = clouds;
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
