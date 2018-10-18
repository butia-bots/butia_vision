#include "object_recognition/yolo_recognition.h"

YoloRecognition::YoloRecognition(ros::NodeHandle _nh) : node_handle(_nh)
{
    readParameters();

    bounding_boxes_sub = node_handle.subscribe(bounding_boxes_topic, bounding_boxes_qs, &YoloRecognition::yoloRecognitionCallback, this);
    recognized_objects_pub = node_handle.advertise<vision_system_msgs::Recognitions>(object_recognition_topic, object_recognition_qs);
    recognized_objects3d_pub = node_handle.advertise<vision_system_msgs::Recognitions3D>(object_recognition3d_topic, object_recognition3d_qs);
    recognized_people_pub = node_handle.advertise<vision_system_msgs::Recognitions>(people_detection_topic, people_detection_qs);
    image2world_client = node_handle.serviceClient<vision_system_msgs::Image2World>(image2world_client_service);
}

bool YoloRecognition::recognitions2Recognitions3D(vision_system_msgs::Recognitions& recognitions, vision_system_msgs::Recognitions3D& recognitions3d)
{
    //tem que gerar uma mensagem
    image2world_srv.request.recognitions = recognitions;
    if(!image2world_client.call(image2world_srv)) {
        ROS_ERROR("Failed to call image2world service");
        return false;
    }

    recognitions3d.image_header = recognitions.image_header;
    recognitions3d.recognition_header = recognitions.recognition_header;

    std::vector<geometry_msgs::PoseWithCovariance> poses;
    std::vector<vision_system_msgs::Description> &objects = recognitions.descriptions; 
    std::vector<vision_system_msgs::Description3D> &descriptions = recognitions3d.descriptions; 
    poses = image2world_srv.response.poses;


    std::vector<geometry_msgs::PoseWithCovariance>::iterator it;
    std::vector<vision_system_msgs::Description>::iterator jt;

    geometry_msgs::Point point;

    for(it = poses.begin(), jt = objects.begin(); it!=poses.end() && jt!=objects.end(); it++, jt++) {
        vision_system_msgs::Description3D description;

        point = it->pose.position;
        std::cout<<"<"<<jt->label_class<<", "<<point.x<<", "<<point.y<<", "<<point.z<<">"<<std::endl;

        description.label_class = jt->label_class;
        description.probability = jt->probability;
        description.pose = *(it);

        descriptions.push_back(description);
    }

    return true;
}

void YoloRecognition::yoloRecognitionCallback(darknet_ros_msgs::BoundingBoxes bbs)
{
    ROS_INFO("Image ID: %d", bbs.image_header.seq);
    std::vector<vision_system_msgs::Description> objects;
    std::vector<vision_system_msgs::Description> people;

    std::vector<darknet_ros_msgs::BoundingBox> bounding_boxes = bbs.bounding_boxes;
    std::vector<darknet_ros_msgs::BoundingBox>::iterator it;

    for(it = bounding_boxes.begin() ; it != bounding_boxes.end() ; it++) {
        if(it->Class == person_identifier) {
            vision_system_msgs::Description person;
            person.label_class = person_identifier;
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

        if(recognitions2Recognitions3D(pub_object_msg, pub_object3D_msg)){
            recognized_objects3d_pub.publish(pub_object3D_msg);
        }

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

    node_handle.param("/object_recognition/publishers/object_recognition3d/topic", object_recognition3d_topic, std::string("/vision_system/or/object_recognition3d"));
    node_handle.param("/object_recognition/publishers/object_recognition3d/queue_size", object_recognition3d_qs, 1);

    node_handle.param("/object_recognition/publishers/people_detection/topic", people_detection_topic, std::string("/vision_system/or/people_detection"));
    node_handle.param("/object_recognition/publishers/people_detection/queue_size", people_detection_qs, 1);

    node_handle.param("/object_recognition/services/image2world/service", image2world_client_service, std::string("/vision_system/iw/image2world"));

    node_handle.param("/object_recognition/person/identifier", person_identifier, std::string("person"));
}
