#!/usr/bin/env python

from mask_rcnn_ros.msg import Result as InstancesResult
from butia_vision_msgs.msg import Description, Recognitions
from butia_vision_msgs.srv import ListClasses, ListClassesResponse
from std_msgs.msg import Header
import rospy


class MaskRCNNSegmentation():
    def __init__(self):
        self.readParameters()
        self.instance_results_sub = rospy.Subscriber(self.instance_results_topic, InstancesResult, self.instanceSegmentationCallback, queue_size=self.instance_segmentation_qs)
        self.recognized_objects_pub = rospy.Publisher(self.object_recognition_topic, Recognitions, queue_size=self.object_recognition_qs)
        self.recognized_people_pub = rospy.Publisher(self.people_detection_topic, Recognitions, queue_size=self.people_detection_qs)
        self.object_list_updated_pub = rospy.Publisher(self.object_list_updated_topic, Header, queue_size=self.object_list_updated_qs)

        self.list_objects_server = rospy.Service(self.list_objects_service, ListClasses, self.getObjectList)

        self.object_list_updated_header = Header()
        self.object_list_updated_header.stamp = rospy.get_rostime().now
        self.object_list_updated_header.frame_id = "objects_list"

        self.object_list_updated_pub.publish(self.object_list_updated_header)

    def instanceSegmentationCallback(self, instance_results):
        rospy.loginfo('Image ID: ' + str(instance_results.header.seq))
        objects = []
        people = []
        for i, (bb, class_id, class_name, score, mask) in enumerate(zip(instance_results.boxes, instance_results.class_ids, instance_results.class_names, instance_results.scores, instance_results.masks)):
            if class_name in self.possible_classes['people'] and score >= self.threshold:
                person = Description()
                person.label_class = "person/{}".format(class_name)
                person.probability = score
                person.bounding_box.minX = bb.x_offset
                person.bounding_box.minY = bb.y_offset
                person.bounding_box.width = bb.width
                person.bounding_box.height = bb.height
                person.mask = mask
                people.append(person)
            
            elif class_name in [val for sublist in self.possible_classes.values() for val in sublist] and score >= self.threshold:
                object_d = Description()
                index = 0
                i = 0
                for value in self.possible_classes.values():
                    if class_name in value:
                        index = i
                    i += 1
                object_d.label_class = list(self.possible_classes.keys())[index] + '/' + class_name
                object_d.probability = score
                object_d.bounding_box.minX = bb.x_offset
                object_d.bounding_box.minY = bb.y_offset
                object_d.bounding_box.width = bb.width
                object_d.bounding_box.height = bb.height
                objects.append(object_d)

        objects_msg = Recognitions()
        people_msg = Recognitions()

        if len(objects) > 0:
            objects_msg.header = instance_results.header
            objects_msg.image_header = instance_results.header
            objects_msg.descriptions = objects
            self.recognized_objects_pub.publish(objects_msg)

        if len(people) > 0:
            people_msg.header = instance_results.header
            people_msg.image_header = instance_results.header
            people_msg.descriptions = people
            self.recognized_people_pub.publish(people_msg)


    def getObjectList(self, req):
        objects_list = []
        for key, value in self.possible_classes.items():
            for elem in value:
                objects_list.append(key + '/' + elem)
        return ListClassesResponse(objects_list)

    def readParameters(self):
        self.instance_results_topic = rospy.get_param("/instance_segmentation/subscribers/instance_results/topic", "/mask_rcnn_ros/results")
        self.instance_segmentation_qs = rospy.get_param("/instance_segmentation/subscribers/instance_results/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("/instance_segmentation/publishers/object_recognition/topic", "/butia_vision/or/object_recognition")
        self.object_recognition_qs = rospy.get_param("/instance_segmentation/publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("/instance_segmentation/publishers/people_detection/topic", "/butia_vision/or/people_detection")
        self.people_detection_qs = rospy.get_param("/instance_segmentation/publishers/people_detection/queue_size", 1)

        self.object_list_updated_topic = rospy.get_param("/instance_segmentation/publishers/object_list_updated/topic", "/butia_vision/or/object_list_updated")
        self.object_list_updated_qs = rospy.get_param("/instance_segmentation/publishers/object_list_updated/queue_size", 1)

        self.list_objects_service = rospy.get_param("/instance_segmentation/servers/list_objects/service", "/butia_vision/or/list_objects")

        self.threshold = rospy.get_param("/instance_segmentation/threshold", 0.5)
        
        self.possible_classes = dict(rospy.get_param("/instance_segmentation/possible_classes"))