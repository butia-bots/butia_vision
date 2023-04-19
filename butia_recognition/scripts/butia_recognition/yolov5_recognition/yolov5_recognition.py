#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import ros_numpy

from butia_recognition import BaseRecognition, ifState

import torch
import numpy as np
import os
from copy import copy
import cv2

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D

torch.set_num_threads(1)

class YoloV5Recognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)

        self.readParameters()

        self.colors = dict([(k, np.random.randint(low=0, high=256, size=(3,)).tolist()) for k in self.classes])

        self.loadModel()
        self.initRosComm()

    def initRosComm(self):
        self.debug_publisher = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.object_recognition_publisher = rospy.Publisher(self.object_recognition_topic, Recognitions2D, queue_size=self.object_recognition_qs)
        self.people_detection_publisher = rospy.Publisher(self.people_detection_topic, Recognitions2D, queue_size=self.people_detection_qs)
        super().initRosComm(callbacks_obj=self)

    def serverStart(self, req):
        self.loadModel()
        return super().serverStart(req)

    def serverStop(self, req):
        self.unLoadModel()
        return super().serverStop(req)

    def loadModel(self):
        self.model = torch.hub.load(os.path.join(self.pkg_path, 'include', 'yolov5'), 'custom', path=os.path.join(self.pkg_path, 'config', 'yolov5_network_config', 'weights', self.model_file), autoshape=True, source='local')
        self.model.eval()
        self.model.conf = self.threshold
        print('Done loading model!')

    def unLoadModel(self):
        del self.model

    @ifState
    def callback(self, *args):
        img = None
        points = None
        if len(args):
            img = args[0]
            points = args[1]

        with torch.no_grad():

            rospy.loginfo('Image ID: ' + str(img.header.seq))

            #cv_img = np.flip(ros_numpy.numpify(img),2)
            cv_img = ros_numpy.numpify(img)
            results = self.model(cv_img)
  
            debug_img = copy(cv_img)

            bbs_l = results.pandas().xyxy[0]

            objects_recognition = Recognitions2D()
            h = Header()
            h.seq = self.seq
            self.seq += 1
            h.stamp = rospy.Time.now()
            objects_recognition.header = h
            objects_recognition.image_rgb = copy(img)
            objects_recognition.points = copy(points)

            people_recognition = copy(objects_recognition)

            description_header = img.header
            description_header.seq = 0
            for i in range(len(bbs_l)):
                if int(bbs_l['class'][i]) >= len(self.classes):
                    continue

                label_class = self.classes[int(bbs_l['class'][i])]

                description = Description2D()
                description.header = copy(description_header)
                description.type = Description2D.DETECTION
                description.id = description.header.seq
                description.score = bbs_l['confidence'][i]
                size = int(bbs_l['xmax'][i] - bbs_l['xmin'][i]), int(bbs_l['ymax'][i] - bbs_l['ymin'][i])
                description.bbox.center.x = int(bbs_l['xmin'][i]) + int(size[0]/2)
                description.bbox.center.y = int(bbs_l['ymin'][i]) + int(size[1]/2)
                description.bbox.size_x = size[0]
                description.bbox.size_y = size[1]

                if ('people' in self.classes and label_class in self.classes_by_category['people'] or 'people' in self.classes and label_class == 'people') and bbs_l['confidence'][i] >= self.threshold:

                    description.label = 'people' + '/' + label_class
                    people_recognition.descriptions.append(description)

                elif (label_class in [val for sublist in self.classes for val in sublist] or label_class in self.classes) and bbs_l['confidence'][i] >= self.threshold:
                    index = None
                    j = 0
                    for value in self.classes_by_category.values():
                        if label_class in value:
                            index = j
                        j += 1
                    description.label = self.classes[5] + '/' + label_class if index is not None else label_class

                    objects_recognition.descriptions.append(description)

                debug_img = cv2.rectangle(debug_img, (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), (int(bbs_l['xmax'][i]), int(bbs_l['ymax'][i])), self.colors[label_class])
                debug_img = cv2.putText(debug_img, label_class, (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=self.colors[label_class])
                description_header.seq += 1
            
            self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))

            if len(objects_recognition.descriptions) > 0:
                self.object_recognition_publisher.publish(objects_recognition)

            if len(people_recognition.descriptions) > 0:
                self.people_detection_publisher.publish(people_recognition)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = rospy.get_param("~publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("~publishers/people_detection/topic", "/butia_vision/br/people_detection")
        self.people_detection_qs = rospy.get_param("~publishers/people_detection/queue_size", 1)

        self.threshold = rospy.get_param("~threshold", 0.5)

        self.classes_by_category = dict(rospy.get_param("~classes_by_category", {}))

        self.model_file = rospy.get_param("~model_file", "larc2021_go_and_get_it.pt")

        super().readParameters()

if __name__ == '__main__':
    rospy.init_node('yolov5_recognition_node', anonymous = True)

    yolo = YoloV5Recognition()

    rospy.spin()
