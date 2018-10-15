#include "object_recognition/yolo_recognition.h"

YoloRecognition::YoloRecognition(ros::NodeHandle _nh) : node_handle(_nh)
{
    readParameters();

    image2world_client = node_handle.serviceClient<vision_system_msgs::Image2World>("/vision_system/iw/image2world");

    bounding_boxes_sub = node_handle.subscribe(bounding_boxes_topic, bounding_boxes_qs, &YoloRecognition::yoloRecognitionCallback, this);
    recognized_objects_pub = node_handle.advertise<vision_system_msgs::Recognitions>(object_recognition_topic, object_recognition_qs);
    recognized_people_pub = node_handle.advertise<vision_system_msgs::Recognitions>(people_detection_topic, people_detection_qs);
}

void YoloRecognition::yoloRecognitionCallback(darknet_ros_msgs::BoundingBoxes bbs)
{
    people.clear();
    objects.clear();

    std::vector<darknet_ros_msgs::BoundingBox> bounding_boxes = bbs.bounding_boxes;
    std::vector<darknet_ros_msgs::BoundingBox>::iterator it;

    for(it = bounding_boxes.begin() ; it != bounding_boxes.end() ; it++) {
        if(it->Class == person_identifier) {
            vision_system_msgs::Description person;
            person.probability = it->probability;
            person.bounding_box.minX = it->xmin;
            person.bounding_box.minY = it->ymin;
            person.bounding_box.width = it->xmax - it->xmin;
            person.bounding_box.height = it->ymax - it->ymin;
            people.push_back(person);
        }
        else {
            vision_system_msgs::Description object;
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
        pub_object_msg.descriptions = objects;
        recognized_objects_pub.publish(pub_object_msg);
        //test
        std::vector<sensor_msgs::PointCloud> clouds;

        vision_system_msgs::Image2World image_srv;
        image_srv.request.recognitions = pub_object_msg;
        image2world_client.call(image_srv);

        clouds = image_srv.response.clouds;

        std::vector<sensor_msgs::PointCloud>::iterator it;
        std::vector<vision_system_msgs::Description>::iterator jt;

        geometry_msgs::Point32 point;

        for(it = clouds.begin(), jt = objects.begin() ; it!=clouds.end() && jt!=objects.end() ; it++, jt++) {
            point = it->points[it->points.size()/2];
            std::cout<<"<"<<jt->label_class<<", "<<point.x<<", "<<point.y<<", "<<point.z<<">"<<std::endl;
        }
        //end_test
    }

    if(people.size() > 0) {
        pub_people_msg.image_header = bbs.image_header;
        pub_people_msg.recognition_header = bbs.header;
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

    node_handle.param("/object_recognition/person/identifier", person_identifier, std::string("person"));
}
