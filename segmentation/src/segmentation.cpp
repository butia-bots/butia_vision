#include "segmentation.hpp"


//----------------------------------Image Segmenter's Functions----------------------------------
ImageSegmenter::ImageSegmenter(ros::NodeHandle _nh) : node_handle(_nh), segmentable_depth = 0 {
    readParameters();
    
    dif = (int *)malloc(3 * sizeof(int));
    dif[0] = -1;
    dif[1] = 0;
    dif[2] = 1;

    service = node_handle.advertiseService(param_segmentation_service, &ImageSegmenter::segment, this);
}

void ImageSegmenter::filterImage(cv::Mat &image)
{
	cv:medianBlur(image, image, 5);
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


void ImageSegmenter::createMaskHistogram() {
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
                    verifyStateHistogram(r, c);
            }
        }
    }

    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask(r, c) == 2)
                mask(r, c) = 0;
        }
    }

    filterImage(mask);

    histogram.clear();
    histogram_class_limits.clear();
    histogram_size = initial_histogram_size;
}


bool ImageSegmenter::verifyStateHistogram(int r, int c) {
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
	mask(r, c) = VERIFYING;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if ((dif[i] == 0) && (dif[j] == 0))
                    continue;
                if ((r + dif[i] >= 0) && (r + dif[i] < mask.rows) && (c + dif[j] >= 0) && (c + dif[j] < mask.cols)) {
                    if (mask(r + dif[i], c + dif[j]) != VERIFYING) {	
                        if (verifyStateHistogram(r + dif[i], c + dif[j]) == true) {
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

void ImageSegmenter::createMaskMedianFull() {
    std::vector<int> med;

    for(int r = 0 ; r < cropped_initial_depth_image.rows ; r++) {
        for(int c = 0 ; c < cropped_initial_depth_image.cols ; c++) {
            if(cropped_initial_depth_image(i, j) <= 0 || cropped_initial_depth_image(r, c) > 5001) continue;
            med.push_back(cropped_initial_depth_image(r, c));
        }    
    }

    std::sort(med.begin(), med.end());

    if(med.size()%2 == 0)
        segmentable_depth =  (med[med.size()/2] + med[(med.size()/2) - 1])/2;
    else
        segmentable_depth = med[(med.size()/2) - 1]);


    for (int r = 0; r < cropped_initial_depth_image.rows; r++) {
        for (int c = 0; c < cropped_initial_depth_image.cols; c++) {
            if (mask(r, c) == NOT_VERIFIED)
                verifyStateMedianFull(r, c);
        }
    }

    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask(r, c) == 2)
                mask(r, c) = 0;
        }
    }

    filterImage(mask);
}

bool ImageSegmenter::verifyStateMedianFull(int r, int c) {
    if (mask(r, c) == TRUTH)
        return true;
    if (mask(r, c) == LIE)
        return false;

    if ((cropped_initial_depth_image(r, c) <= lower_histogram_limit) || (cropped_initial_depth_image(r, c) >= upper_histogram_limit)) {
        mask(r, c) = LIE;
        return false;
    }
    
    if ((cropped_initial_depth_image(r, c) >= segmentable - 100) && (cropped_initial_depth_image(r, c) < segmentable + 100)) {
        mask(r, c) = TRUTH;
        return true;
    }

    bool answer = false;
    for (int i = 1; i <= left_class_limit && answer == false; i++) {
        if (position_of_max_value - i >= 0) {
            if ((cropped_initial_depth_image(r, c) >= segmentable_depth - median_center_threshold*(i+1)) && (cropped_initial_depth_image(r, c) <= segmentable_depth - median_center_threshold*i))
                answer = true;
        }
    }
    for (int i = 1; i <= right_class_limit && answer == false; i++) {
        if (position_of_max_value + i < histogram_size) {
            if ((cropped_initial_depth_image(r, c) >= segmentable_depth + median_center_threshold*i) && (cropped_initial_depth_image(r, c) <= segmentable_depth + median_center_threshold*(i+1)))
               answer = true;
        }
    }

    if (answer == true) {
	mask(r, c) = VERIFYING;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if ((dif[i] == 0) && (dif[j] == 0))
                    continue;
                if ((r + dif[i] >= 0) && (r + dif[i] < mask.rows) && (c + dif[j] >= 0) && (c + dif[j] < mask.cols)) {
                    if (mask(r + dif[i], c + dif[j]) != VERIFYING) {	
                        if (verifyStateMedianFull(r + dif[i], c + dif[j]) == true) {
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

void ImageSegmenter::createMaskMedianCenter() {
    int deriv[median_center_kernel_size];
    for(int i = 0, i < median_center_kernel_size ; i++){
        deriv[i] = i - (median_center_kernel_size/2);
    }
    std::vector<int> med;

    int r_i = cropped_initial_depth_image.rows/2;
    int c_i = cropped_initial_depth_image.cols/2;

    for(int i = 0 ; i < median_center_kernel_size ; i++) {
        for(int j = 0 ; j < median_center_kernel_size ; j++) {
            if((r_i + i < 0 || r_i + i >= cropped_initial_depth_image.rows) || (c_i + j < 0 || c_i + j >= cropped_initial_depth_image.cols)) continue;
            if(cropped_initial_depth_image(r_i + i, c_i + j) <= lower_limit || cropped_initial_depth_image(r_i + i, c_i + j) >= upper_limit) continue;
            med.push_back(cropped_initial_depth_image(r_i + i, c_i + j));
        }    
    }

    std::sort(med.begin(), med.end());

    if(med.size()%2 == 0)
        segmentable_depth =  (med[med.size()/2] + med[(med.size()/2) - 1])/2;
    else
        segmentable_depth = med[(med.size()/2) - 1]);

    for (int r = 0; r < cropped_initial_depth_image.rows; r++) {
        for (int c = 0; c < cropped_initial_depth_image.cols; c++) {
            if (mask(r, c) == NOT_VERIFIED)
                verifyStateMedianCenter(r, c);
        }
    }

    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask(r, c) == 2)
                mask(r, c) = 0;
        }
    }

    filterImage(mask);
}

bool ImageSegmenter::verifyStateMedianCenter(int r, int c) {
    if (mask(r, c) == TRUTH)
        return true;
    if (mask(r, c) == LIE)
        return false;

    if ((cropped_initial_depth_image(r, c) <= lower_histogram_limit) || (cropped_initial_depth_image(r, c) >= upper_histogram_limit)) {
        mask(r, c) = LIE;
        return false;
    }
    
    if ((cropped_initial_depth_image(r, c) >= segmentable - 100) && (cropped_initial_depth_image(r, c) < segmentable + 100)) {
        mask(r, c) = TRUTH;
        return true;
    }

    bool answer = false;
    for (int i = 1; i <= left_class_limit && answer == false; i++) {
        if (position_of_max_value - i >= 0) {
            if ((cropped_initial_depth_image(r, c) >= segmentable_depth - median_center_threshold*(i+1)) && (cropped_initial_depth_image(r, c) <= segmentable_depth - median_center_threshold*i))
                answer = true;
        }
    }
    for (int i = 1; i <= right_class_limit && answer == false; i++) {
        if (position_of_max_value + i < histogram_size) {
            if ((cropped_initial_depth_image(r, c) >= segmentable_depth + median_center_threshold*i) && (cropped_initial_depth_image(r, c) <= segmentable_depth + median_center_threshold*(i+1)))
               answer = true;
        }
    }

    if (answer == true) {
	mask(r, c) = VERIFYING;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if ((dif[i] == 0) && (dif[j] == 0))
                    continue;
                if ((r + dif[i] >= 0) && (r + dif[i] < mask.rows) && (c + dif[j] >= 0) && (c + dif[j] < mask.cols)) {
                    if (mask(r + dif[i], c + dif[j]) != VERIFYING) {	
                        if (verifyStateMedianCenter(r + dif[i], c + dif[j]) == true) {
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
    node_handle.param("/segmentation/servers/image_segmentation/service", param_segmentation_service, std::string("/vision_system/seg/image_segmentation"));

    node_handle.param("/segmentation/model_id", model_id, std::string("median_full"));
    node_handle.param("/segmentation/lower_limit", lower_limit, 0);
    node_handle.param("/segmentation/upper_limit", upper_limit, 5001);
    node_handle.param("/segmentation/left_class_limit", left_class_limit, 1);
    node_handle.param("/segmentation/right_class_limit", right_class_limit, 1);

    node_handle.param("/segmentation/histogram/size", histogram_size, 20);
    node_handle.param("/segmentation/historam/bounding_box_threshold", bounding_box_threshold, (float)0.35);
    node_handle.param("/segmentation/historam/decrease_factor", histogram_decrease_factor, (float)2.0);

    node_handle.param("/segmentation/median_full/threshold", median_full_threshold, 100);

    node_handle.param("/segmentation/median_center/kernel_size", median_center_kernel_size, 5);
    node_handle.param("/segmentation/median_center/threshold", median_center_threshold, 100);
}
//-----------------------------------------------------------------------------------------------
