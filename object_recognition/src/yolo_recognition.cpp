#include "object_recognition/yolo_recognition.h"

YoloRecognition::YoloRecognition(ros::NodeHandle _nh) : node_handle(_nh)
{
    readParameters();

    bounding_boxes_sub = node_handle.subscribe(bounding_boxes_topic, bounding_boxes_qs, &YoloRecognition::yoloRecognitionCallback, this);

    recognized_objects_pub = node_handle.advertise<butia_vision_msgs::Recognitions>(object_recognition_topic, object_recognition_qs);
    recognized_people_pub = node_handle.advertise<butia_vision_msgs::Recognitions>(people_detection_topic, people_detection_qs);
    object_list_updated_pub = node_handle.advertise<std_msgs::Header>(object_list_updated_topic, object_list_updated_qs);

    list_objects_server = node_handle.advertiseService(list_objects_service, &YoloRecognition::getObjectList, this);

    object_list_updated_header.stamp = ros::Time::now();
    object_list_updated_header.frame_id = "objects_list";

    object_list_updated_pub.publish(object_list_updated_header);
}

bool YoloRecognition::getObjectList(butia_vision_msgs::ListClasses::Request &req, butia_vision_msgs::ListClasses::Response &res)
{
    //res.classes = possible_classes;
    return true;
}

void YoloRecognition::yoloRecognitionCallback(darknet_ros_msgs::BoundingBoxes bbs)
{
    ROS_INFO("Image ID: %d", bbs.image_header.seq);

    /*for(std::map<std::string, std::vector<std::string> >::const_iterator it = possible_classes.begin(); it != possible_classes.end(); ++it)
    {
        std::cout << it->first << " : ";
        for(std::vector<std::string>::iterator jt = it->second.begin(); jt != it->second.end(); ++jt)
            std::cout << *jt << " ";
        std::cout<<std::endl;
    }*/

    std::vector<butia_vision_msgs::Description> objects;
    std::vector<butia_vision_msgs::Description> people;

    std::vector<darknet_ros_msgs::BoundingBox> bounding_boxes = bbs.bounding_boxes;
    std::vector<darknet_ros_msgs::BoundingBox>::iterator it;

    for(it = bounding_boxes.begin() ; it != bounding_boxes.end() ; it++) {
        if(it->Class == person_identifier && it->probability >= threshold) {
            butia_vision_msgs::Description person;
            person.label_class = person_identifier;
            person.probability = it->probability;
            person.bounding_box.minX = it->xmin;
            person.bounding_box.minY = it->ymin;
            person.bounding_box.width = it->xmax - it->xmin;
            person.bounding_box.height = it->ymax - it->ymin;
            people.push_back(person);
        }
        else if(it->probability >= threshold) {
            //if(std::find(possible_classes.begin(), possible_classes.end(), std::string(it->Class)) != possible_classes.end()) {   
                butia_vision_msgs::Description object;
                object.label_class = it->Class;
                object.probability = it->probability;
                object.bounding_box.minX = it->xmin;
                object.bounding_box.minY = it->ymin;
                object.bounding_box.width = it->xmax - it->xmin;
                object.bounding_box.height = it->ymax - it->ymin;
                objects.push_back(object);
            //}
        }
    }

    std::vector<butia_vision_msgs::Description>::iterator jt;

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

    node_handle.param("/object_recognition/publishers/object_recognition/topic", object_recognition_topic, std::string("/butia_vision/or/object_recognition"));
    node_handle.param("/object_recognition/publishers/object_recognition/queue_size", object_recognition_qs, 1);

    node_handle.param("/object_recognition/publishers/people_detection/topic", people_detection_topic, std::string("/butia_vision/or/people_detection"));
    node_handle.param("/object_recognition/publishers/people_detection/queue_size", people_detection_qs, 1);

    node_handle.param("/object_recognition/publishers/object_list_updated/topic", object_list_updated_topic, std::string("/butia_vision/or/object_list_updated"));
    node_handle.param("/object_recognition/publishers/object_list_updated/queue_size", object_list_updated_qs, 1);

    node_handle.param("/object_recognition/servers/list_objects/service", list_objects_service, std::string("/butia_vision/or/list_objects"));

    node_handle.param("/object_recognition/person_identifier", person_identifier, std::string("person"));

    node_handle.param("/object_recognition/threshold", threshold, (float)0.5);
    
    //node_handle.getParam("/object_recognition/possible_classes", possible_classes);
    

}
