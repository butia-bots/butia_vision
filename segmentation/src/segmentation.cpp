#include "segmentation.hpp"



//----------------------------------Image Segmenter's Functions----------------------------------
ImageSegmenter::ImageSegmenter(ros::NodeHandle _nh) : node_handle(_nh), histogram_size(20), lower_histogram_limit(1), upper_histogram_limit(5001), histogram_decrease_factor(2), bounding_box_threshold(0.25) {
    range = (float *)malloc(2 * sizeof(float));
    histogram_range = (const float **)malloc(sizeof(float *));
    range[0] = lower_histogram_limit;
    range[1] = upper_histogram_limit;
    histogram_range[0] = range;

    d = (int *)malloc(3 * sizeof(int));
    d[0] = -1;
    d[1] = 0;
    d[2] = 1;

    service = node_handle.advertiseService("/vision_system/seg/image_segmentation", &ImageSegmenter::segment, this);
}


bool ImageSegmenter::segment(vision_system_msgs::ImageSegmentation::Request &req, vision_system_msgs::ImageSegmentation::Response &res) {
    sensor_msgs::Image::ConstPtr constptr_initial_rgb_image(new sensor_msgs::Image(req.initial_rgbd_image.rgb));
    sensor_msgs::Image::ConstPtr constptr_initial_depth_image(new sensor_msgs::Image(req.initial_rgbd_image.depth));

    readImage(constptr_initial_rgb_image, mat_initial_rgb_image);
    readImage(constptr_initial_depth_image, mat_initial_depth_image);

    cropImage(mat_initial_depth_image, req.bounding_box);
    cropImage(mat_initial_rgb_image, req.bounding_box);

    cv::Mat_<cv::Vec3b> mat_segmented_rgb_image(mat_initial_depth_image.rows, mat_initial_depth_image.cols, CV_8UC3);
    mask = cv::Mat_<uint8_t>(mat_initial_depth_image.rows, mat_initial_depth_image.cols, CV_8UC1);

    createMask();

    ROS_INFO("Mask created!");

    for (int r = 0; r < mat_segmented_rgb_image.rows; r++) {
        for (int c = 0; c < mat_segmented_rgb_image.cols; c++) {
            mat_segmented_rgb_image(r, c)[0] = mat_initial_rgb_image(r, c)[0] * mask(r, c);
            mat_segmented_rgb_image(r, c)[1] = mat_initial_rgb_image(r, c)[1] * mask(r, c);
            mat_segmented_rgb_image(r, c)[2] = mat_initial_rgb_image(r, c)[2] * mask(r, c);
        }
    }

    cv::namedWindow("Segmented image", 1);
    cv::imshow("Segmented image", mat_segmented_rgb_image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return true;
}


void ImageSegmenter::readImage(const sensor_msgs::Image::ConstPtr &msg_image, cv::Mat &image) {
    cv_bridge::CvImageConstPtr cv_image;
    cv_image = cv_bridge::toCvShare(msg_image, msg_image->encoding);
    cv_image->image.copyTo(image);
}


void ImageSegmenter::cropImage(cv::Mat &image, vision_system_msgs::BoundingBox bounding_box) {
    cv::Rect region_of_interest(bounding_box.minX, bounding_box.minY, bounding_box.width, bounding_box.height);
    image = image(region_of_interest);
}


void ImageSegmenter::calculateHistogram() { 
    //Creating the histogram
    for(int r = 0; r < mat_initial_depth_image.rows; r++) {
        uint16_t *it = mat_initial_depth_image.ptr<uint16_t>(r);
        for(int c = 0; c < mat_initial_depth_image.cols; c++, it++) {
                if (*it == 0)
                    continue;
                histogram[(int)(((*it)*histogram_size) / upper_histogram_limit)]++;
        }
    }
}


void ImageSegmenter::getMaxHistogramValue() {
    for (int i = 0; i < histogram.size(); i++) {
        if (histogram[i] > max_histogram_value) {
            max_histogram_value = histogram[i];
            position_of_max_value = i;
        }
    }
}


void ImageSegmenter::createMask() {
    bool object_founded;
    bool validated;
    float initial_sill;
    float accumulated_sill;

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
    state = cv::Mat_<uint8_t>(mask.rows, mask.cols, CV_8UC1);
    if (object_founded == true) {
        ROS_INFO("Object founded!");
        for (int r = 0; r < mat_initial_depth_image.rows; r++) {
            for (int c = 0; c < mat_initial_depth_image.cols; c++) {
                if (state(r, c) == NOT_VERIFIED) {
                    state(r, c) = 1;
                    verifyState(r, c);
                }
            }
        }
    }

    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if (mask(r, c) == 2)
                mask(r, c) = 0;
        }
    }
}


bool ImageSegmenter::verifyState(int r, int c) {
    if (mask(r, c) == TRUTH)
        return true;
    if (mask(r, c) == LIE)
        return false;
    
    //uint16_t *it_depth = mat_initial_depth_image.ptr<uint16_t>(r);
    //uint8_t *it_mask = depth.ptr<uint8_t>(r);
    if ((mat_initial_depth_image(r, c) >= histogram_class_limits[position_of_max_value].first) && (mat_initial_depth_image(r, c) < histogram_class_limits[position_of_max_value].second)) {
        mask(r, c) = TRUTH;
        state(r, c) = 1;
        return true;
    }

    bool r1 = false, r2 = false;
    if (position_of_max_value > 0)
        r1 = (mat_initial_depth_image(r, c) >= histogram_class_limits[position_of_max_value - 1].first) && (mat_initial_depth_image(r, c) < histogram_class_limits[position_of_max_value - 1].second);
    if (position_of_max_value < histogram_size - 1)
        r2 = (mat_initial_depth_image(r, c) >= histogram_class_limits[position_of_max_value + 1].first) && (mat_initial_depth_image(r, c) < histogram_class_limits[position_of_max_value + 1].second);
    if (r1 || r2) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if ((i == 1) && (j == 1))
                    continue;
                if ((r + d[i] >= 0) && (r + d[i] < mask.rows) && (c + d[j] >= 0) && (c + d[j] < mask.cols)) {
                    if (state(r + d[i], c + d[j]) != NOT_VERIFIED) {
                        if (verifyState(r + d[i], c + d[j]) == true) {
                            mask(r, c) = TRUTH;
                            state(r, c) = 1;
                            return true;
                        }
                    }
                }
            }
        }
    }

    mask(r, c) = LIE;
    state(r, c) = 1;
    return false;
}
//-----------------------------------------------------------------------------------------------