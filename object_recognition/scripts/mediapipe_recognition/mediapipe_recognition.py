#!/usr/bin/env python3
import rospy

from std_msgs.msg import Header
import cv2
import cv_bridge
import numpy as np
import rospkg
import os
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description, Recognitions
from butia_vision_msgs.srv import ListClasses, ListClassesResponse
import mediapipe as mp



class MediapipeRecognition():
    def __init__(self):
        self.readParameters()

        self.rospack = rospkg.RosPack()
        print(self.rospack.get_path('object_recognition'))

        self.mp_pose = mp.solutions.pose
        #self.bounding_boxes_sub = rospy.Subscriber(self.bounding_boxes_topic, BoundingBoxes, self.yoloRecognitionCallback, queue_size=self.bounding_boxes_qs)
        self.bridge = cv_bridge.CvBridge()
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.mediapipeRecognitionCallback, queue_size=self.image_qs)

        self.recognized_people_pub = rospy.Publisher(self.people_detection_topic, Recognitions, queue_size=self.people_detection_qs)

    def mediapipeRecognitionCallback(self, img):

        with self.mp_pose.Pose(static_image_mode=self.static_image_mode, min_detection_confidence=self.min_detection_confidence, model_complexity=self.model_complexity) as pose:

            rospy.loginfo('Image ID: ' + str(img.header.seq))

            cv_img = self.bridge.imgmsg_to_cv2(img, desired_encoding='rgb8').copy()

            #results = self.model(torch.tensor(cv_img.reshape((1, 3, 640, 480)).astype(np.float32)).to(self.model.device))
            results = pose.process(cv_img)

            bbs_l = results.pandas().xyxy[0]

            objects = []
            people = []

            for i in range(len(bbs_l)):
                print(bbs_l['name'][i])
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                cv_img = cv2.rectangle(cv_img, (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), (int(bbs_l['xmax'][i]), int(bbs_l['ymax'][i])), colors[bbs_l['name'][i]])
                cv_img = cv2.putText(cv_img, bbs_l['name'][i], (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=colors[bbs_l['name'][i]])
                if bbs_l['name'][i] in dictionary.keys():
                    reference_model = bbs_l['name'][i]
                    bbs_l['name'][i] = dictionary[bbs_l['name'][i]]
                    
                if 'people' in self.possible_classes and bbs_l['name'][i] in self.possible_classes['people'] and bbs_l['confidence'][i] >= self.threshold:
                    person = Description()
                    person.label_class = 'people' + '/' + bbs_l['name'][i]
                    person.reference_model = reference_model
                    person.probability = bbs_l['confidence'][i]
                    person.bounding_box.minX = int(bbs_l['xmin'][i])
                    person.bounding_box.minY = int(bbs_l['ymin'][i])
                    person.bounding_box.width = int(bbs_l['xmax'][i] - bbs_l['xmin'][i])
                    person.bounding_box.height = int(bbs_l['ymax'][i] - bbs_l['ymin'][i])
                    people.append(person)

                elif bbs_l['name'][i] in [val for sublist in self.possible_classes.values() for val in sublist] and bbs_l['confidence'][i] >= self.threshold:
                    object_d = Description()
                    index = 0
                    j = 0
                    for value in self.possible_classes.values():
                        if bbs_l['name'][i] in value:
                            index = j
                        j += 1
                    object_d.reference_model = reference_model
                    object_d.label_class = list(self.possible_classes.keys())[index] + '/' + bbs_l['name'][i]
                    object_d.probability = bbs_l['confidence'][i]
                    object_d.bounding_box.minX = int(bbs_l['xmin'][i])
                    object_d.bounding_box.minY = int(bbs_l['ymin'][i])
                    object_d.bounding_box.width = int(bbs_l['xmax'][i] - bbs_l['xmin'][i])
                    object_d.bounding_box.height = int(bbs_l['ymax'][i]- bbs_l['ymin'][i])
                    objects.append(object_d)

            cv2.imshow('YoloV5', cv_img)
            cv2.waitKey(1)

            objects_msg = Recognitions()
            people_msg = Recognitions()

            if len(objects) > 0:
                #objects_msg.header = bbs.header
                objects_msg.image_header = img.header
                objects_msg.descriptions = objects
                self.recognized_objects_pub.publish(objects_msg)

            if len(people) > 0:
                #people_msg.header = bbs.header
                people_msg.image_header = img.header
                people_msg.descriptions = people
                self.recognized_people_pub.publish(people_msg)
            


    def readParameters(self):
        self.bounding_boxes_topic = rospy.get_param("/object_recognition/subscribers/bounding_boxes/topic", "/darknet_ros/bounding_boxes")
        self.bounding_boxes_qs = rospy.get_param("/object_recognition/subscribers/bounding_boxes/queue_size", 1)

        self.image_topic = rospy.get_param("/object_recognition/subscribers/image/topic", "/butia_vision/bvb/image_rgb_raw")
        self.image_qs = rospy.get_param("/object_recognition/subscribers/image/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("/object_recognition/publishers/object_recognition/topic", "/butia_vision/or/object_recognition")
        self.object_recognition_qs = rospy.get_param("/object_recognition/publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("/object_recognition/publishers/people_detection/topic", "/butia_vision/or/people_detection")
        self.people_detection_qs = rospy.get_param("/object_recognition/publishers/people_detection/queue_size", 1)

        self.object_list_updated_topic = rospy.get_param("/object_recognition/publishers/object_list_updated/topic", "/butia_vision/or/object_list_updated")
        self.object_list_updated_qs = rospy.get_param("/object_recognition/publishers/object_list_updated/queue_size", 1)

        self.list_objects_service = rospy.get_param("/object_recognition/servers/list_objects/service", "/butia_vision/or/list_objects")

        self.threshold = rospy.get_param("/object_recognition/threshold", 0.5)
        
        self.possible_classes = dict(rospy.get_param("/object_recognition/possible_classes"))

        self.model_file = rospy.get_param("/object_recognition/model_file", "larc2021_go_and_get_it.pt")
