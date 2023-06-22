#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import ros_numpy
from butia_recognition import BaseRecognition, ifState
import numpy as np
import os
from copy import copy
import cv2
from ultralytics import YOLO
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D


class YoloV8Recognition(BaseRecognition):
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
        self.model = YOLO("/home/butiabots/Workspace/butia_ws/src/butia_vision/butia_recognition/config/yolov8_network_config/yolov8_lab_objects.pt")
        self.model.conf = self.threshold
        print('Done loading model!')

    def unLoadModel(self):
        del self.model

    @ifState
    def callback(self, *args):
        source_data = self.sourceDataFromArgs(args)

        if 'image_rgb' not in source_data:
            rospy.logwarn('Souce data has no image_rgb.')
            return None
        
        img_rgb = source_data['image_rgb']
        cv_img = ros_numpy.numpify(img_rgb)
        rospy.loginfo('Image ID: ' + str(img_rgb.header.seq))
        
        objects_recognition = Recognitions2D()
        h = Header()
        h.seq = self.seq #id mensagem
        self.seq += 1 #prox id
        h.stamp = rospy.Time.now()

        objects_recognition.header = h
        objects_recognition = BaseRecognition.addSourceData2Recognitions2D(source_data, objects_recognition)
        people_recognition = copy(objects_recognition)
        description_header = img_rgb.header
        description_header.seq = 0

        results = self.model.predict(cv_img)
        boxes_ = results[0].boxes.cpu().numpy()

        if len(results[0].boxes):
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                xyxy_box = list(boxes_[i].xyxy.astype(int)[0])
                
                if int(box.cls) >= len(self.all_classes):
                    continue

                label_class = self.all_classes[int(box.cls)]

                description = Description2D()
                description.header = copy(description_header)
                description.type = Description2D.DETECTION
                description.id = description.header.seq
                description.score = float(box.conf)
                size = int(xyxy_box[2] - xyxy_box[0]), int(xyxy_box[3] - xyxy_box[1])
                description.bbox.center.x = int(xyxy_box[0]) + int(size[0]/2)
                description.bbox.center.y = int(xyxy_box[1]) + int(size[1]/2)
                description.bbox.size_x = size[0]
                description.bbox.size_y = size[1]

                if ('people' in self.all_classes and label_class in self.classes_by_category['people'] or 'people' in self.all_classes and label_class == 'people') and box.conf >= self.threshold:

                    description.label = 'people' + '/' + label_class
                    people_recognition.descriptions.append(description)

                elif (label_class in [val for sublist in self.all_classes for val in sublist] or label_class in self.all_classes) and box.conf >= self.threshold:
                    index = None

                    for value in self.classes_by_category.values():
                        if label_class in value[1]:
                            index = value[0]

                    description.label = index + '/' + label_class if index is not None else label_class
                    objects_recognition.descriptions.append(description)

                debug_img = results[0].plot()
                description_header.seq += 1
            
            self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))

            if len(objects_recognition.descriptions) > 0:
                self.object_recognition_publisher.publish(objects_recognition)

            if len(people_recognition.descriptions) > 0:
                self.people_detection_publisher.publish(people_recognition)       
        else:
            debug_img = results[0].plot()            
            self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = rospy.get_param("~publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("~publishers/people_detection/topic", "/butia_vision/br/people_detection")
        self.people_detection_qs = rospy.get_param("~publishers/people_detection/queue_size", 1)

        self.threshold = rospy.get_param("~threshold", 0.5)


        self.all_classes = list(rospy.get_param("/object_recognition/all_classes"))
        self.classes_by_category = dict(rospy.get_param("~classes_by_category", {}))
        self.model_file = rospy.get_param("~model_file", "yolov8_lab_objects.pt")

        super().readParameters()

if __name__ == '__main__':
    rospy.init_node('yolov8_recognition_node', anonymous = True)

    yolo = YoloV8Recognition()

    rospy.spin()