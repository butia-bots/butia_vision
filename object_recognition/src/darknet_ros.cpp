#include "darknet_ros.h"

DarknetROS::DarknetROS(ros::NodeHandle _nh) : nh(_nh)
{
    loadParameters();

    bounding_boxes_sub = nh.subscribe();
}

void DarknetROS::darknetRosCallback(darknet_ros_msgs::BoundingBoxes bbs)
{
    pub_object_msg = vision_system_msgs::RecognizedObjects();
    pub_people_msg = vision_system_msgs::RecognizedPeople();

    std::vector<vision_system_msgs::ObjectDescription> objects;
    std::vector<vision_system_msgs::PeopleDescription> people;

    std::vector<darknet_ros_msgs::BoundingBox> bounding_boxes = bbs.bounding_boxes;
    std::vector<darknet_ros_msgs::BoundingBox>::iterator it;

    for(it = bounding_boxes.begin() ; it != bounding_boxes.end() ; it++) {
        if(it->Class == "Person") {
            person = vision_system_msgs::PersonDescription();
            person.probability = it->probability;
            person.bounding_box.minX = it->xmin;
            person.bounding_box.minY = it->ymin;
            person.bounding_box.width = it->xmax - it->xmin;
            person.bounding_box.height = it->ymax - it->ymin;
            people.push_back(person);
        }
        else {
            object = vision_system_msgs::ObjectDescription();
            object.label = it->Class;
            object.probability = it->probability;
            object.bounding_box.minX = it->xmin;
            object.bounding_box.minY = it->ymin;
            object.bounding_box.width = it->xmax - it->xmin;
            object.bounding_box.height = it->ymax - it->ymin;
            objects.push_back(*it);
        }
    }

    if(people.size() > 0) {
        pub_people_msg.image_header = bbs.image_header;
        pub_people_msg.recognition_header = bbs.header;
        pub_people_msg.people_description = people;
    }

    if(objects.size() > 0) {

    }
}

void DarknetROS::loadParameters()
{

}
