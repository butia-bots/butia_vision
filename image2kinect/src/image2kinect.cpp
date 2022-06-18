#include "image2kinect/image2kinect.h"

Image2Kinect::Image2Kinect(ros::NodeHandle _nh) : node_handle(_nh), width(0), height(0)
{
    readParameters();

    loadObjectClouds();

    object_recognition_sub = node_handle.subscribe(object_recognition_sub_topic, sub_queue_size, &Image2Kinect::objectRecognitionCallback, this);
    face_recognition_sub = node_handle.subscribe(face_recognition_sub_topic, sub_queue_size, &Image2Kinect::faceRecognitionCallback, this);
    people_detection_sub = node_handle.subscribe(people_detection_sub_topic, sub_queue_size, &Image2Kinect::peopleDetectionCallback, this);
    people_tracking_sub = node_handle.subscribe(people_tracking_sub_topic, sub_queue_size, &Image2Kinect::peopleTrackingCallback, this);

    object_recognition_pub = node_handle.advertise<butia_vision_msgs::Recognitions3D>(object_recognition_pub_topic, pub_queue_size);
    face_recognition_pub = node_handle.advertise<butia_vision_msgs::Recognitions3D>(face_recognition_pub_topic, pub_queue_size);
    people_detection_pub = node_handle.advertise<butia_vision_msgs::Recognitions3D>(people_detection_pub_topic, pub_queue_size);
    people_tracking_pub = node_handle.advertise<butia_vision_msgs::Recognitions3D>(people_tracking_pub_topic, pub_queue_size);

    image_request_client = node_handle.serviceClient<butia_vision_msgs::ImageRequest>(image_request_client_service);
    segmentation_request_client = node_handle.serviceClient<butia_vision_msgs::SegmentationRequest>(segmentation_request_client_service);
}

bool Image2Kinect::points2RGBPoseWithCovariance(PointCloud &points, butia_vision_msgs::BoundingBox &bb, geometry_msgs::PoseWithCovariance &pose, std_msgs::ColorRGBA &color, cv::Mat &mask, PointCloud &object_points)
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

    if(!use_mask && kernel_size > 0) {
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
        min_x = bb.minX <= 0 ? 0 : bb.minX;
        min_y = bb.minY <= 0 ? 0 : bb.minY;

        max_x = min_x + bb.width > points.width - 1 ? points.width - 1 : min_x + bb.width;
        max_y = min_y + bb.height > points.height - 1 ? points.height - 1 : min_y + bb.height; 
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

    object_points.resize(valid_points.size());
    for (auto point : valid_points) {
        object_points.push_back(point);
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

    // if(use_align){
    //     std::vector<std::string> split_label;
    //     boost::algorithm::split(split_label, label, boost::is_any_of("/"));
    //     if (category2dataset.count(split_label.back()) > 0) {
    //         std::string translated_label = category2dataset[split_label.back()];
    //         PointCloud::Ptr shared_points(new PointCloud());
    //         for (auto &point : valid_points) {
    //             shared_points->points.push_back(point);
    //         }
    //         if (object_clouds.count(translated_label) > 0)
    //         {
    //             PointCloud::Ptr voxelized_cloud(new PointCloud);
    //             pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    //             voxel_grid.setLeafSize(0.01, 0.01, 0.01);
    //             voxel_grid.setInputCloud(shared_points);
    //             voxel_grid.filter(*voxelized_cloud);
    //             Eigen::Vector4f centroid;
    //             pcl::compute3DCentroid(*voxelized_cloud, centroid);
    //             Eigen::Matrix4f center_cloud_transform = Eigen::Matrix4f::Identity();
    //             center_cloud_transform(0, 3) = -centroid(0);
    //             center_cloud_transform(1, 3) = -centroid(1);
    //             center_cloud_transform(2, 3) = -centroid(2);
    //             PointCloud::Ptr centered_cloud(new PointCloud());
    //             pcl::transformPointCloud(*voxelized_cloud, *centered_cloud, center_cloud_transform);
    //             std::cout << center_cloud_transform << std::endl;
    //             pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    //             icp.setInputSource(centered_cloud);
    //             icp.setInputTarget(object_clouds[translated_label]);
    //             icp.setMaximumIterations(100);
    //             PointCloud::Ptr aligned_object_cloud(new PointCloud());
    //             icp.align(*aligned_object_cloud);
    //             Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
    //             Eigen::Matrix4f final_transform = -center_cloud_transform * icp_transform;
    //             std::cout << final_transform << std::endl;
    //             Eigen::Quaternionf rotation(final_transform.block<3,3>(0, 0));
    //             mean_position.x = (double)final_transform(0, 3);
    //             mean_position.y = (double)final_transform(1, 3);
    //             mean_position.z = (double)final_transform(2, 3);
    //             orientation.w = (double)rotation.w();
    //             orientation.x = (double)rotation.x();
    //             orientation.y = (double)rotation.y();
    //             orientation.z = (double)rotation.z();
    //         }
    //     }
    // }
    return true;
}


bool Image2Kinect::robustPoseEstimation(PointCloud &points, butia_vision_msgs::BoundingBox &bb, geometry_msgs::PoseWithCovariance &pose, cv::Mat &mask, std::string label)
{
    geometry_msgs::Point &position = pose.pose.position;
    geometry_msgs::Quaternion &orientation = pose.pose.orientation;

    bool use_mask = false;
    if(mask.rows*mask.cols != 0) use_mask = true;

    int min_x, min_y, max_x, max_y;

    if(!use_mask && kernel_size > 0) {
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
        min_x = bb.minX <= 0 ? 0 : bb.minX;
        min_y = bb.minY <= 0 ? 0 : bb.minY;

        max_x = min_x + bb.width > points.width - 1 ? points.width - 1 : min_x + bb.width;
        max_y = min_y + bb.height > points.height - 1 ? points.height - 1 : min_y + bb.height; 
    }

    PointCloudT::Ptr scene(new PointCloudT);
    FeatureCloudT::Ptr object_features (new FeatureCloudT);
    FeatureCloudT::Ptr scene_features (new FeatureCloudT);
    PointCloudT::Ptr object_aligned (new PointCloudT);

    for(int r = min_y; r <= max_y ; r++) {
        for(int c = min_x; c <= max_x ; c++) {
            pcl::PointXYZRGB point = points.at(c, r);
            float depth = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
            if((use_mask && mask.at<int>(r - min_y,c - min_x) != 0) || !use_mask) {
                if(depth <= max_depth && depth != 0){  
                    PointNT pn;
                    pn.x = point.x;
                    pn.y = point.y;
                    pn.z = point.z;
                    scene->push_back(pn);
                }
            }
        }
    }

    if(scene->width*scene->height == 0) {
        ROS_ERROR("Alignment Error!");
        return false;
    }

    pcl::VoxelGrid<PointNT> voxel_grid;
    voxel_grid.setLeafSize(0.01, 0.01, 0.01);
    voxel_grid.setInputCloud(scene);
    voxel_grid.filter(*scene);
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*scene, centroid);
    Eigen::Matrix4f center_cloud_transform = Eigen::Matrix4f::Identity();
    center_cloud_transform(0, 3) = -centroid(0);
    center_cloud_transform(1, 3) = -centroid(1);
    center_cloud_transform(2, 3) = -centroid(2);
    pcl::transformPointCloudWithNormals(*scene, *scene, center_cloud_transform);

    if (object_clouds.count(label) > 0 && scene->size() > 10)
    {
        pcl::NormalEstimationOMP<PointNT,PointNT> nest;
        nest.setRadiusSearch(0.02);
        nest.setInputCloud(scene);
        nest.compute(*scene);

        std::cout<<"Reference model: "<<object_clouds[label]->size()<<" points."<<std::endl;
        std::cout<<"Scene Cloud: "<<scene->size()<<" points."<<std::endl;

        // if(label == "002_master_chef_can"){
        //     pcl::io::savePCDFileASCII ("/home/igormaurell/object_pre.pcd", *object_clouds[label]);
        // }

        // pcl::IterativeClosestPoint<PointNT, PointNT> align;
        // align.setInputSource(scene);
        // align.setInputTarget(object_clouds[label]);
        // align.setMaximumIterations(100);
        // align.align(*object_aligned);

        FeatureEstimationT fest;
        fest.setRadiusSearch (0.025);
        fest.setInputCloud (object_clouds[label]);
        fest.setInputNormals (object_clouds[label]);
        fest.compute (*object_features);
        fest.setInputCloud (scene);
        fest.setInputNormals (scene);
        fest.compute (*scene_features);

        pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align;
        align.setInputSource (object_clouds[label]);
        align.setSourceFeatures (object_features);
        align.setInputTarget (scene);
        align.setTargetFeatures (scene_features);
        align.setMaximumIterations (20000); // Number of RANSAC iterations
        align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
        align.setCorrespondenceRandomness (5); // Number of nearest features to use
        align.setSimilarityThreshold (0.9f); // Polygonal edge length similarity threshold
        align.setMaxCorrespondenceDistance (2.5f * 0.01); // Inlier threshold
        align.setInlierFraction (0.25f);

        align.align (*object_aligned);
        
        if (align.hasConverged ())
        {
            ROS_INFO("Alignment Done!");
            Eigen::Matrix4f transformation = align.getFinalTransformation ();

            // if(label == "002_master_chef_can"){
            //     pcl::io::savePCDFileASCII ("/home/igormaurell/object.pcd", *object_aligned);
            //     pcl::io::savePCDFileASCII ("/home/igormaurell/scene.pcd", *scene);
            // }'

            Eigen::Matrix4f final_transform = center_cloud_transform.inverse() * transformation;

            Eigen::Quaternionf rotation(final_transform.block<3,3>(0, 0));
            position.x = (double)final_transform(0, 3);
            position.y = (double)final_transform(1, 3);
            position.z = (double)final_transform(2, 3);
            orientation.w = (double)rotation.w();
            orientation.x = (double)rotation.x();
            orientation.y = (double)rotation.y();
            orientation.z = (double)rotation.z();
            
            return true;
        }
        ROS_ERROR("Alignment Error!");
    }
    else {
        std::string error = "There is no Reference Model with name: ";
        error += label + "!";
        ROS_ERROR(error.c_str());
    }

    position.x = -center_cloud_transform(0, 3);
    position.y = -center_cloud_transform(1, 3);
    position.z = -center_cloud_transform(2, 3);
    orientation.w = 0;
    orientation.x = 0;
    orientation.y = 0;
    orientation.z = 1;
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

    butia_vision_msgs::SegmentationRequest segmentation_srv;
    segmentation_srv.request.initial_rgbd_image = rgbd_image;
    segmentation_srv.request.descriptions = descriptions;
    if (!segmentation_request_client.call(segmentation_srv)) {
        ROS_ERROR("Failed to call segmentation service");
       return;
    }

    //here, it is possible to apply point cloud filters, as: cut-off, voxel grid, statistical and others

    std::vector<butia_vision_msgs::Description>::iterator it;

    std::vector<sensor_msgs::Image> &segmented_rgb_images = segmentation_srv.response.segmented_rgb_images;
    std::vector<sensor_msgs::Image>::iterator jt;
  
    std::vector<butia_vision_msgs::Description3D> &descriptions3d = recognitions3d.descriptions;


    for(it = descriptions.begin(); it != descriptions.end(); it++) {
        butia_vision_msgs::Description3D description3d;
        description3d.label_class = it->label_class;
        description3d.probability = it->probability;
        std_msgs::ColorRGBA &color =  description3d.color;
        geometry_msgs::PoseWithCovariance &pose = description3d.pose;
        cv::Mat mask;
        PointCloud object_points;
        if(it->mask.height * it->mask.width != 0) {
            sensor_msgs::Image::ConstPtr mask_const_ptr( new sensor_msgs::Image(it->mask));
            readImage(mask_const_ptr, mask);
        } else {
            cv::Rect roi;
            roi.x = (*it).bounding_box.minX;
            roi.y = (*it).bounding_box.minY;
            roi.width = (*it).bounding_box.width;
            roi.height = (*it).bounding_box.height;
            segmented_depth_image = depth(roi);
            mask = segmented_depth_image;
        }

        sensor_msgs::Image::ConstPtr rgb_const_ptr( new sensor_msgs::Image(*jt));
        readImage(rgb_const_ptr, segmented_rgb_image);

        bool success = false;
        if(use_align) {
            success = robustPoseEstimation(points, (*it).bounding_box, pose, mask, it->reference_model);
        }
        else {
            success = points2RGBPoseWithCovariance(points, (*it).bounding_box, pose, color, mask, object_points);
            pcl::toROSMsg(object_points, description3d.points);
        }

        if(success) {
            descriptions3d.push_back(description3d);
        }
        
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
        //q.setRPY(0, 0, 0);
        q.setW(it->pose.pose.orientation.w);
        q.setX(it->pose.pose.orientation.x);
        q.setY(it->pose.pose.orientation.y);
        q.setZ(it->pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), recognitions3d.image_header.frame_id,
                                              it->label_class + std::to_string(current_rec[it->label_class]) + "_detect"));
    }
}

void Image2Kinect::objectRecognitionCallback(butia_vision_msgs::Recognitions recognitions)
{
    butia_vision_msgs::Recognitions3D recognitions3d;
    segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    object_recognition_pub.publish(recognitions3d);
}

void Image2Kinect::faceRecognitionCallback(butia_vision_msgs::Recognitions recognitions)
{
    butia_vision_msgs::Recognitions3D recognitions3d;
    segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    face_recognition_pub.publish(recognitions3d);
}

void Image2Kinect::peopleDetectionCallback(butia_vision_msgs::Recognitions recognitions)
{
    butia_vision_msgs::Recognitions3D recognitions3d;
    segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    people_detection_pub.publish(recognitions3d);
}

void Image2Kinect::peopleTrackingCallback(butia_vision_msgs::Recognitions recognitions)
{
    butia_vision_msgs::Recognitions3D recognitions3d;
    segmentation_model_id = "median_full";
    recognitions2Recognitions3d(recognitions, recognitions3d);
    people_tracking_pub.publish(recognitions3d);
}

void Image2Kinect::loadObjectClouds()
{
    std::string objects_dir = ros::package::getPath("image2kinect") + "/data";
    for (boost::filesystem::directory_iterator iter(objects_dir); iter != boost::filesystem::directory_iterator(); ++iter)
    {
        if (iter->path().filename().extension() == ".pcd")
        {
            std::string label = iter->path().filename().stem().string();
            //ROS_INFO(label.c_str());
            PointCloudT::Ptr object_cloud(new PointCloudT());
            pcl::io::loadPCDFile<PointNT>(iter->path().string(), *object_cloud);
            pcl::VoxelGrid<PointNT> voxel_grid;
            voxel_grid.setLeafSize(0.01, 0.01, 0.01);
            voxel_grid.setInputCloud(object_cloud);
            voxel_grid.filter(*object_cloud);
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*object_cloud, centroid);
            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
            transform(0, 3) = -centroid(0);
            transform(1, 3) = -centroid(1);
            transform(2, 3) = -centroid(2);
            pcl::transformPointCloudWithNormals(*object_cloud, *object_cloud, transform);
            object_clouds.insert(std::make_pair(label, object_cloud));
            object_transforms.insert(std::make_pair(label, transform));
        }
    }
}
/*
void Image2Kinect::refinePose(const PointCloud::Ptr &points, geometry_msgs::PoseWithCovariance &pcov, std::string label)
{
    if (object_clouds.count(label) > 0)
    {
        PointCloud::Ptr voxelized_cloud(new PointCloud);
        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
        voxel_grid.setLeafSize(0.01, 0.01, 0.01);
        voxel_grid.setInputCloud(points);
        voxel_grid.filter(*voxelized_cloud);
        geometry_msgs::Pose &pose = pcov.pose;
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*voxelized_cloud, centroid);
        Eigen::Matrix4f center_cloud_transform = Eigen::Matrix4f::Identity();
        center_cloud_transform(0, 3) = -centroid(0);
        center_cloud_transform(1, 3) = -centroid(1);
        center_cloud_transform(2, 3) = -centroid(2);
        PointCloud::Ptr centered_cloud(new PointCloud());
        pcl::transformPointCloud(*voxelized_cloud, *centered_cloud, center_cloud_transform);
        std::cout << center_cloud_transform << std::endl;
        pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
        icp.setInputSource(centered_cloud);
        icp.setInputTarget(object_clouds[label]);
        icp.setMaximumIterations(1);
        PointCloud::Ptr aligned_object_cloud(new PointCloud());
        icp.align(*aligned_object_cloud);
        Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
        Eigen::Matrix4f final_transform = center_cloud_transform * icp_transform;
        std::cout << final_transform << std::endl;
        Eigen::Quaternionf rotation(final_transform.block<3,3>(0, 0));
        pose.position.x = (double)final_transform(0, 3);
        pose.position.y = (double)final_transform(1, 3);
        pose.position.z = (double)final_transform(2, 3);
        pose.orientation.w = (double)rotation.w();
        pose.orientation.x = (double)rotation.x();
        pose.orientation.y = (double)rotation.y();
        pose.orientation.z = (double)rotation.z();
    }
}
*/
void Image2Kinect::readParameters()
{
    node_handle.param("/image2kinect/subscribers/queue_size", sub_queue_size, 5);
    node_handle.param("/image2kinect/subscribers/object_recognition/topic", object_recognition_sub_topic, std::string("/butia_vision/or/object_recognition"));
    node_handle.param("/image2kinect/subscribers/face_recognition/topic", face_recognition_sub_topic, std::string("/butia_vision/fr/face_recognition"));
    node_handle.param("/image2kinect/subscribers/people_detection/topic", people_detection_sub_topic, std::string("/butia_vision/or/people_detection"));
    node_handle.param("/image2kinect/subscribers/people_tracking/topic", people_tracking_sub_topic, std::string("/butia_vision/pt/people_tracking"));

    node_handle.param("/image2kinect/publishers/queue_size", pub_queue_size, 5);
    node_handle.param("/image2kinect/publishers/object_recognition/topic", object_recognition_pub_topic, std::string("/butia_vision/or/object_recognition3d"));
    node_handle.param("/image2kinect/publishers/face_recognition/topic", face_recognition_pub_topic, std::string("/butia_vision/fr/face_recognition3d"));
    node_handle.param("/image2kinect/publishers/people_detection/topic", people_detection_pub_topic, std::string("/butia_vision/or/people_detection3d"));
    node_handle.param("/image2kinect/publishers/people_tracking/topic", people_tracking_pub_topic, std::string("/butia_vision/pt/people_tracking3d"));
    
    node_handle.param("/image2kinect/clients/image_request/service", image_request_client_service, std::string("/butia_vision/bvb/image_request"));
    node_handle.param("/image2kinect/clients/segmentation_request/service", segmentation_request_client_service, std::string("/butia_vision/seg/image_segmentation"));

    node_handle.param("/image2kinect/segmentation_threshold", segmentation_threshold, (float)0.2);
    node_handle.param("/image2kinect/max_depth", max_depth, 2000);
    node_handle.param("/image2kinect/publish_tf", publish_tf, true);
    node_handle.param("/image2kinect/kernel_size", kernel_size, 5);
    node_handle.param("/image2kinect/use_align", use_align, true);
}
