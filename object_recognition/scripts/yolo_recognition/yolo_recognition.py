#! /usr/bin/env python
import rospy

from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from butia_vision_msgs.msg import Description, Recognitions
from butia_vision_msgs.srv import ListClasses, ListClassesResponse

class YoloRecognition():
    def __init__(self):
        self.readParameters()

        self.bounding_boxes_sub = rospy.Subscriber(self.bounding_boxes_topic, BoundingBoxes, self.yoloRecognitionCallback, queue_size=self.bounding_boxes_qs)
        
        self.recognized_objects_pub = rospy.Publisher(self.object_recognition_topic, Recognitions, queue_size=self.object_recognition_qs)
        self.recognized_people_pub = rospy.Publisher(self.people_detection_topic, Recognitions, queue_size=self.people_detection_qs)
        self.object_list_updated_pub = rospy.Publisher(self.object_list_updated_topic, Header, queue_size=self.object_list_updated_qs)

        self.list_objects_server = rospy.Service(self.list_objects_service, ListClasses, self.getObjectList)

        self.object_list_updated_header = Header()
        self.object_list_updated_header.stamp = rospy.get_rostime().now
        self.object_list_updated_header.frame_id = "objects_list"

        self.object_list_updated_pub.publish(self.object_list_updated_header)
    
    def getObjectList(self, req):
        objects_list = []
        for key, value in self.possible_classes.items():
            for elem in value:
                objects_list.append(key + '/' + elem)
        return ListClassesResponse(objects_list)

    def yoloRecognitionCallback(self, bbs):
        rospy.loginfo('Image ID: ' + str(bbs.image_header.seq))

        bbs_l = bbs.bounding_boxes

        objects = []
        people = []

        for bb in bbs_l:
            if bb.Class in self.possible_classes['people'] and bb.probability >= self.threshold:
                person = Description()
                person.label_class = 'people' + '/' + bb.Class
                person.probability = bb.probability
                person.bounding_box.minX = bb.xmin
                person.bounding_box.minY = bb.ymin
                person.bounding_box.width = bb.xmax - bb.xmin
                person.bounding_box.height = bb.ymax - bb.ymin
                people.append(person)

            elif bb.Class in [val for sublist in self.possible_classes.values() for val in sublist] and bb.probability >= self.threshold:
                object_d = Description()
                index = 0
                i = 0
                for value in self.possible_classes.values():
                    if bb.Class in value:
                        index = i
                    i += 1
                object_d.label_class = list(self.possible_classes.keys())[index] + '/' + bb.Class
                object_d.probability = bb.probability
                object_d.bounding_box.minX = bb.xmin
                object_d.bounding_box.minY = bb.ymin
                object_d.bounding_box.width = bb.xmax - bb.xmin
                object_d.bounding_box.height = bb.ymax - bb.ymin
                objects.append(object_d)

        objects_msg = Recognitions()
        people_msg = Recognitions()

        if len(objects) > 0:
            objects_msg.header = bbs.header
            objects_msg.image_header = bbs.image_header
            objects_msg.descriptions = objects
            self.recognized_objects_pub.publish(objects_msg)

        if len(people) > 0:
            people_msg.header = bbs.header
            people_msg.image_header = bbs.image_header
            people_msg.descriptions = people
            self.recognized_people_pub.publish(people_msg)
            


    def readParameters(self):
        self.bounding_boxes_topic = rospy.get_param("/object_recognition/subscribers/bounding_boxes/topic", "/darknet_ros/bounding_boxes")
        self.bounding_boxes_qs = rospy.get_param("/object_recognition/subscribers/bounding_boxes/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("/object_recognition/publishers/object_recognition/topic", "/butia_vision/or/object_recognition")
        self.object_recognition_qs = rospy.get_param("/object_recognition/publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("/object_recognition/publishers/people_detection/topic", "/butia_vision/or/people_detection")
        self.people_detection_qs = rospy.get_param("/object_recognition/publishers/people_detection/queue_size", 1)

        self.object_list_updated_topic = rospy.get_param("/object_recognition/publishers/object_list_updated/topic", "/butia_vision/or/object_list_updated")
        self.object_list_updated_qs = rospy.get_param("/object_recognition/publishers/object_list_updated/queue_size", 1)

        self.list_objects_service = rospy.get_param("/object_recognition/servers/list_objects/service", "/butia_vision/or/list_objects")

        self.threshold = rospy.get_param("/object_recognition/threshold", 0.5)
        
        self.possible_classes = dict(rospy.get_param("/object_recognition/possible_classes"))