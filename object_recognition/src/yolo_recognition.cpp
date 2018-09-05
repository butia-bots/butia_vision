#include "yolo_recognition.h"

YoloRecognition::YoloRecognition(ros::NodeHandle _nh) : node_handle(_nh)
{
    readParameters();

    bounding_boxes_sub = node_handle.subscribe(bounding_boxes_topic, bounding_boxes_qs, &YoloRecognition::yoloRecognitionCallback, this);
    recognized_objects_pub = node_handle.advertise<vision_system_msgs::RecognizedObjects>(object_recognition_topic, object_recognition_qs);
    recognized_people_pub = node_handle.advertise<vision_system_msgs::RecognizedPeople>(people_recognition_topic, people_recognition_qs);
}

void YoloRecognition::yoloRecognitionCallback(darknet_ros_msgs::BoundingBoxes bbs)
{
    vision_system_msgs::RecognizedObjects pub_object_msg;
    vision_system_msgs::RecognizedPeople pub_people_msg;

    std::vector<vision_system_msgs::ObjectDescription> objects;
    std::vector<vision_system_msgs::PersonDescription> people;

    std::vector<darknet_ros_msgs::BoundingBox> bounding_boxes = bbs.bounding_boxes;
    std::vector<darknet_ros_msgs::BoundingBox>::iterator it;

    for(it = bounding_boxes.begin() ; it != bounding_boxes.end() ; it++) {
        if(it->Class == "Person") {
            vision_system_msgs::PersonDescription person;
            person.probability = it->probability;
            person.bounding_box.minX = it->xmin;
            person.bounding_box.minY = it->ymin;
            person.bounding_box.width = it->xmax - it->xmin;
            person.bounding_box.height = it->ymax - it->ymin;
            people.push_back(person);
        }
        else {
            vision_system_msgs::ObjectDescription object;
            object.label_class = it->Class;
            object.probability = it->probability;
            object.bounding_box.minX = it->xmin;
            object.bounding_box.minY = it->ymin;
            object.bounding_box.width = it->xmax - it->xmin;
            object.bounding_box.height = it->ymax - it->ymin;
            objects.push_back(object);
        }
    }

    if(objects.size() > 0) {
        pub_object_msg.image_header = bbs.image_header;
        pub_object_msg.recognition_header = bbs.header;
        pub_object_msg.objects_description = objects;
        recognized_objects_pub.publish(pub_object_msg);
    }

    if(people.size() > 0) {
        pub_people_msg.image_header = bbs.image_header;
        pub_people_msg.recognition_header = bbs.header;
        pub_people_msg.people_description = people;
        recognized_people_pub.publish(pub_people_msg);
    }
}

void YoloRecognition::readParameters()
{
    node_handle.param("/subscribers/bounding_boxes/topic", bounding_boxes_topic, std::string("/darknet_ros/bounding_boxes"));
    node_handle.param("/subscribers/bounding_boxes/queue_size", bounding_boxes_qs, 1);

    node_handle.param("/publishers/object_recognition/topic", object_recognition_topic, std::string("/vision_system/or/recognized_objects"));
    node_handle.param("/publishers/object_recognition/queue_size", object_recognition_qs, 1);

    node_handle.param("/publishers/people_recognition/topic", people_recognition_topic, std::string("/vision_system/or/recognized_people"));
    node_handle.param("/publishers/people_recognition/queue_size", people_recognition_qs, 1);
}
