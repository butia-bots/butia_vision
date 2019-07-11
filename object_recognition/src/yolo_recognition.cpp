#include "object_recognition/yolo_recognition.h"

YoloRecognition::YoloRecognition(ros::NodeHandle _nh) : node_handle(_nh)
{
    readParameters();

    bounding_boxes_sub = node_handle.subscribe(bounding_boxes_topic, bounding_boxes_qs, &YoloRecognition::yoloRecognitionCallback, this);

    recognized_objects_pub = node_handle.advertise<vision_system_msgs::Recognitions>(object_recognition_topic, object_recognition_qs);
    recognized_people_pub = node_handle.advertise<vision_system_msgs::Recognitions>(people_detection_topic, people_detection_qs);
    object_list_updated_pub = node_handle.advertise<std_msgs::Header>(object_list_updated_topic, object_list_updated_qs);

    list_objects_server = node_handle.advertiseService(list_objects_service, &YoloRecognition::getObjectList, this);

    object_list_updated_header.stamp = ros::Time::now();
    object_list_updated_header.frame_id = "objects_list";

    object_list_updated_pub.publish(object_list_updated_header);
}

bool YoloRecognition::getObjectList(vision_system_msgs::ListClasses::Request &req, vision_system_msgs::ListClasses::Response &res)
{
    res.classes = possible_classes;
    return true;
}

void YoloRecognition::yoloRecognitionCallback(darknet_ros_msgs::BoundingBoxes bbs)
{
    ROS_INFO("Image ID: %d", bbs.image_header.seq);

    /*std::vector<vision_system_msgs::Description> drinks;
    std::vector<vision_system_msgs::Description> snacks;
    std::vector<vision_system_msgs::Description> fruits;
    std::vector<vision_system_msgs::Description> daily;*/

    std::vector<vision_system_msgs::Description> objects;
    std::vector<vision_system_msgs::Description> people;

    std::vector<darknet_ros_msgs::BoundingBox> bounding_boxes = bbs.bounding_boxes;
    std::vector<darknet_ros_msgs::BoundingBox>::iterator it;

    for(it = bounding_boxes.begin() ; it != bounding_boxes.end() ; it++) {
        if(it->Class == person_identifier && it->probability >= threshold) {
            vision_system_msgs::Description person;
            person.label_class = person_identifier;
            person.probability = it->probability;
            person.bounding_box.minX = it->xmin;
            person.bounding_box.minY = it->ymin;
            person.bounding_box.width = it->xmax - it->xmin;
            person.bounding_box.height = it->ymax - it->ymin;
            people.push_back(person);
        }
        else if(it->probability >= threshold) {
            if(std::find(possible_classes.begin(), possible_classes.end(), std::string(it->Class)) != possible_classes.end()) {   
                vision_system_msgs::Description object;
                object.label_class = it->Class;
                object.probability = it->probability;
                object.bounding_box.minX = it->xmin;
                object.bounding_box.minY = it->ymin;
                object.bounding_box.width = it->xmax - it->xmin;
                object.bounding_box.height = it->ymax - it->ymin;
                objects.push_back(object);
                /*if(std::find(DRINKS.begin(), DRINKS.end(), std::string(it->Class)) != DRINKS.end())
                    drinks.push_back(object);
                else if(std::find(SNACKS.begin(), SNACKS.end(), std::string(it->Class)) != SNACKS.end())
                    snacks.push_back(object);
                else if(std::find(FRUITS.begin(), FRUITS.end(), std::string(it->Class)) != FRUITS.end())
                    fruits.push_back(object);
                else if(std::find(DAILY.begin(), DAILY.end(), std::string(it->Class)) != DAILY.end())
                    daily.push_back(object);*/
            }
        }
    }

    std::vector<vision_system_msgs::Description>::iterator jt;

    /*ROS_INFO("DRINKS: ");
    for(jt = drinks.begin() ; jt != drinks.end() ; jt++) {
        ROS_INFO("%s", jt->label_class);
    }
    ROS_INFO("\n\nSNACKS: ");
    for(jt = snacks.begin() ; jt != snacks.end() ; jt++) {
        ROS_INFO("%s", jt->label_class);
    }
    ROS_INFO("\n\nFRUITS: ");
    for(jt = fruits.begin() ; jt != fruits.end() ; jt++) {
        ROS_INFO("%s", jt->label_class);    
    }
    ROS_INFO("\n\nDAILY: ");
    for(jt = daily.begin() ; jt != daily.end() ; jt++) {
        ROS_INFO("%s", jt->label_class);
    }*/

    if(objects.size() > 0) {
        pub_object_msg.header = bbs.header;
        pub_object_msg.image_header = bbs.image_header;
        pub_object_msg.descriptions = objects;
        recognized_objects_pub.publish(pub_object_msg);
    }

    if(people.size() > 0) {
        pub_people_msg.header = bbs.header;
        pub_people_msg.image_header = bbs.image_header;
        pub_people_msg.descriptions = people;
        recognized_people_pub.publish(pub_people_msg);
    }
}

void YoloRecognition::readParameters()
{
    node_handle.param("/object_recognition/subscribers/bounding_boxes/topic", bounding_boxes_topic, std::string("/darknet_ros/bounding_boxes"));
    node_handle.param("/object_recognition/subscribers/bounding_boxes/queue_size", bounding_boxes_qs, 1);

    node_handle.param("/object_recognition/publishers/object_recognition/topic", object_recognition_topic, std::string("/vision_system/or/object_recognition"));
    node_handle.param("/object_recognition/publishers/object_recognition/queue_size", object_recognition_qs, 1);

    node_handle.param("/object_recognition/publishers/people_detection/topic", people_detection_topic, std::string("/vision_system/or/people_detection"));
    node_handle.param("/object_recognition/publishers/people_detection/queue_size", people_detection_qs, 1);

    node_handle.param("/object_recognition/publishers/object_list_updated/topic", object_list_updated_topic, std::string("/vision_system/or/object_list_updated"));
    node_handle.param("/object_recognition/publishers/object_list_updated/queue_size", object_list_updated_qs, 1);

    node_handle.param("/object_recognition/servers/list_objects/service", list_objects_service, std::string("/vision_system/or/list_objects"));

    node_handle.param("/object_recognition/person_identifier", person_identifier, std::string("person"));

    node_handle.param("/object_recognition/threshold", threshold, (float)0.5);
    
    node_handle.param("object_recognition/possible_classes", possible_classes, DEFAULT_CLASS_LIST);
}
