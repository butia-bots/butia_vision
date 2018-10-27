#include "segmentation.hpp"



//----------------------------------Image Segmenter's Functions----------------------------------
ImageSegmenter::ImageSegmenter(ros::NodeHandle _nh) : node_handle(_nh) {
    readParameters();
    
    dif = (int *)malloc(3 * sizeof(int));
    dif[0] = -1;
    dif[1] = 0;
    dif[2] = 1;

    service = node_handle.advertiseService(param_segmentation_service, &ImageSegmenter::segment, this);
}


bool ImageSegmenter::segment(vision_system_msgs::SegmentationRequest::Request &req, vision_system_msgs::SegmentationRequest::Response &res) {
    sensor_msgs::Image::ConstPtr constptr_initial_rgb_image(new sensor_msgs::Image(req.initial_rgbd_image.rgb));
    sensor_msgs::Image::ConstPtr constptr_initial_depth_image(new sensor_msgs::Image(req.initial_rgbd_image.depth));

    readImage(constptr_initial_rgb_image, mat_initial_rgb_image);
    readImage(constptr_initial_depth_image, mat_initial_depth_image);

	std::vector<vision_system_msgs::Description> &descriptions = req.descriptions;
	std::vector<vision_system_msgs::Description>::iterator it;

    std::vector<sensor_msgs::Image> &vector_segmented_msg_image = res.segmented_rgb_images;

	cv::Mat_<cv::Vec3b> mat_segmented_image;

	cv_bridge::CvImage ros_segmented_rgb_image;
    sensor_msgs::Image ros_segmented_msg_image;
	ros_segmented_rgb_image.encoding = req.initial_rgbd_image.rgb.encoding;

    for (it = descriptions.begin(); it != descriptions.end(); it++) {
        cropImage(mat_initial_depth_image, it->bounding_box, cropped_initial_depth_image);
        cropImage(mat_initial_rgb_image, it->bounding_box, cropped_initial_rgb_image);

        mat_segmented_image = cv::Mat_<cv::Vec3b>(cv::Size(cropped_initial_depth_image.cols, cropped_initial_depth_image.rows), CV_8UC3);
        mask = cv::Mat_<uint8_t>(cv::Size(cropped_initial_depth_image.cols, cropped_initial_depth_image.rows), CV_8UC1);

        createMask();

        for (int r = 0; r < mask.rows; r++) {
            for (int c = 0; c < mask.cols; c++) {
                mat_segmented_image(r, c)[0] = cropped_initial_rgb_image(r, c)[0] * mask(r, c);
                mat_segmented_image(r, c)[1] = cropped_initial_rgb_image(r, c)[1] * mask(r, c);
                mat_segmented_image(r, c)[2] = cropped_initial_rgb_image(r, c)[2] * mask(r, c);
            }
        }

        ros_segmented_rgb_image.image = mat_segmented_image;
        ros_segmented_rgb_image.toImageMsg(ros_segmented_msg_image);
        vector_segmented_msg_image.push_back(ros_segmented_msg_image);
    }

    return true;
}


void ImageSegmenter::readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image) {
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(msg_image, msg_image->encoding);
    cv_image->image.copyTo(image);
}


void ImageSegmenter::cropImage(cv::Mat &image, vision_system_msgs::BoundingBox bounding_box, cv::Mat &destiny) {
    cv::Rect region_of_interest(bounding_box.minX, bounding_box.minY, bounding_box.width, bounding_box.height);
    destiny = image(region_of_interest);
}


void ImageSegmenter::calculateHistogram() { 
    //Creating the histogram
    for(int r = 0; r < mat_initial_depth_image.rows; r++) {
        for(int c = 0; c < mat_initial_depth_image.cols; c++) {
            if ((cropped_initial_depth_image(r, c) <= lower_histogram_limit) || (cropped_initial_depth_image(r, c) >= upper_histogram_limit))
                continue;
            histogram[(int)((cropped_initial_depth_image(r, c)*histogram_size) / upper_histogram_limit)]++;
        }
    }
}


void ImageSegmenter::getMaxHistogramValue() {
    max_histogram_value = 0;
    position_of_max_value = 0;

    for (int i = 0; i < histogram_size; i++) {
        if (histogram[i] > max_histogram_value) {
            max_histogram_value = histogram[i];
            position_of_max_value = i;
        }
    }
}


void ImageSegmenter::createMask() {
    bool object_founded;
    float initial_sill;
    float accumulated_sill;
    int initial_histogram_size = histogram_size;

    //Calculating the histogram
    do {
        histogram.resize(histogram_size);
        histogram_class_limits.resize(histogram_size);

        //Filling the limits vector
        initial_sill = upper_histogram_limit / histogram_size;
        accumulated_sill = 0;
        for (int i = 0; i < histogram_size; i++) {
            histogram_class_limits[i].first = (int)accumulated_sill;
            accumulated_sill = accumulated_sill + initial_sill;
            histogram_class_limits[i].second = (int)accumulated_sill;
        }

        calculateHistogram();
        getMaxHistogramValue();
        if ((float)max_histogram_value < histogram_size * bounding_box_threshold) {
            ROS_ERROR("Couldn't found any object!");
            object_founded = false;
            histogram_size = (int)(histogram_size / histogram_decrease_factor);
            histogram.clear();
            histogram_class_limits.clear();
        } else
            object_founded = true;
    } while ((object_founded == false) && (histogram_size >= 8));

    //Creating the mask
    if (object_founded == true) {
        for (int r = 0; r < cropped_initial_depth_image.rows; r++) {
            for (int c = 0; c < cropped_initial_depth_image.cols; c++) {
                if (mask(r, c) == NOT_VERIFIED)
                    verifyState(r, c);
            }
        }
    }

    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask(r, c) == 2)
                mask(r, c) = 0;
        }
    }

    histogram.clear();
    histogram_class_limits.clear();
    histogram_size = initial_histogram_size;
}


bool ImageSegmenter::verifyState(int r, int c) {
    if (mask(r, c) == TRUTH)
        return true;
    if (mask(r, c) == LIE)
        return false;

    if ((cropped_initial_depth_image(r, c) <= lower_histogram_limit) || (cropped_initial_depth_image(r, c) >= upper_histogram_limit)) {
        mask(r, c) = LIE;
        return false;
    }
    
    if ((cropped_initial_depth_image(r, c) >= histogram_class_limits[position_of_max_value].first) && (cropped_initial_depth_image(r, c) < histogram_class_limits[position_of_max_value].second)) {
        mask(r, c) = TRUTH;
        return true;
    }

    bool answer = false;
    for (int i = 1; i <= left_class_limit && answer == false; i++) {
        if (position_of_max_value - i >= 0) {
            if ((cropped_initial_depth_image(r, c) >= histogram_class_limits[position_of_max_value - i].first) && (cropped_initial_depth_image(r, c) <= histogram_class_limits[position_of_max_value - i].second))
                answer = true;
        }
    }
    for (int i = 1; i <= right_class_limit && answer == false; i++) {
        if (position_of_max_value + i < histogram_size) {
            if ((cropped_initial_depth_image(r, c) >= histogram_class_limits[position_of_max_value + i].first) && (cropped_initial_depth_image(r, c) <= histogram_class_limits[position_of_max_value + i].second))
                answer = true;
        }
    }

    if (answer == true) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if ((dif[i] == 0) && (dif[j] == 0))
                    continue;
                if ((r + dif[i] >= 0) && (r + dif[i] < mask.rows) && (c + dif[j] >= 0) && (c + dif[j] < mask.cols)) {
                    if (mask(r + dif[i], c + dif[j]) != VERIFYING) {
                        mask(r, c) = VERIFYING;
                        if (verifyState(r + dif[i], c + dif[j]) == true) {
                            mask(r, c) = TRUTH;
                            return true;
                        }
                    }
                }
            }
        }
    }

    mask(r, c) = LIE;
    return false;
}


void ImageSegmenter::readParameters() {
    node_handle.param("/segmentation/service/image_segmentation/service", param_segmentation_service, std::string("/vision_system/seg/image_segmentation"));

    node_handle.param("/segmentation/parameters/histogram/size", histogram_size, 20);
    node_handle.param("/segmentation/parameters/historam/upper_limit", upper_histogram_limit, 5001);
    node_handle.param("/segmentation/parameters/historam/lower_limit", lower_histogram_limit, 0);
    node_handle.param("/segmentation/parameters/historam/left_class_limit", left_class_limit, 2);
    node_handle.param("/segmentation/parameters/historam/right_class_limit", right_class_limit, 2);
    node_handle.param("/segmentation/parameters/historam/bounding_box_threshold", bounding_box_threshold, (float)0.35);
    node_handle.param("/segmentation/parameters/historam/decrease_factor", histogram_decrease_factor, (float)2.0);
}
//-----------------------------------------------------------------------------------------------
