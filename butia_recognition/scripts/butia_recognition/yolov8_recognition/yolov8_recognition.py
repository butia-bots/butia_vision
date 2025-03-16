#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import ros_numpy
from butia_recognition import BaseRecognition, ifState
import numpy as np
import os
from copy import copy
import cv2
from ultralytics import YOLO, YOLOWorld
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from butia_vision_msgs.msg import Description2D, Recognitions2D
from butia_vision_msgs.srv import SetClass, SetClassRequest, SetClassResponse
import torch
import rospkg

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
        self.set_class_server = rospy.Service(self.set_class_service, SetClass, self.handle_set_class)
        super().initRosComm(callbacks_obj=self)

    def handle_set_class(self, req: SetClassRequest):
        if isinstance(self.model, YOLOWorld):
            if req.class_name != '' and req.class_name not in self.all_classes:
                self.all_classes.append(req.class_name)
            self.model.set_classes(self.all_classes)
        if req.confidence > 0.0:
            self.threshold = req.confidence
        return SetClassResponse()
    
    def serverStart(self, req):
        self.loadModel()
        return super().serverStart(req)

    def serverStop(self, req):
        self.unLoadModel()
        return super().serverStop(req)

    def loadModel(self): 
        self.model = YOLO(self.model_file)
        self.model.conf = self.threshold
        if isinstance(self.model, YOLOWorld):
            self.model.set_classes(self.all_classes)
        print('Done loading model!')

    def unLoadModel(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None

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
        h.frame_id = img_rgb.header.frame_id

        objects_recognition.header = h
        objects_recognition = BaseRecognition.addSourceData2Recognitions2D(source_data, objects_recognition)
        people_recognition = copy(objects_recognition)
        description_header = img_rgb.header
        description_header.seq = 0

        results = self.model.predict(cv_img[:, :, ::-1], conf=self.threshold, verbose=True)
        boxes_ = results[0].boxes.cpu().numpy()
        if results[0].masks is not None:
            masks_ = results[0].masks.cpu().numpy()
        else:
            masks_ = None

        if len(results[0].boxes):
            for i in range(len(results[0].boxes)):
                box = results[0].boxes[i]
                xyxy_box = list(boxes_[i].xyxy.astype(int)[0])
                if boxes_[i].id != None:
                    global_id = int(boxes_[i].id)
                else:
                    global_id = -1
                
                if int(box.cls) >= len(self.all_classes):
                    continue

                label_class = self.all_classes[int(box.cls)]
                max_size = self.max_sizes[label_class]

                description = Description2D()
                description.header = copy(description_header)
                if masks_ is not None:
                    description.type = Description2D.DETECTION
                else:
                    description.type = Description2D.INSTANCE_SEGMENTATION
                    description.mask = ros_numpy.msgify(Image, masks_[i])
                description.id = description.header.seq
                description.global_id = global_id
                description.score = float(box.conf)
                description.max_size = Vector3(*max_size)
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

                    for value in self.classes_by_category.items():
                        if label_class in value[1]:
                            index = value[0]

                    description.label = index + '/' + label_class if index is not None else label_class
                    objects_recognition.descriptions.append(description)

                debug_img = results[0].plot()[:, :, ::-1]
                description_header.seq += 1
            
            self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))
            objects_recognition.image_set_of_marks = ros_numpy.msgify(Image, debug_img, 'rgb8')

            if len(objects_recognition.descriptions) > 0:
                self.object_recognition_publisher.publish(objects_recognition)

            if len(people_recognition.descriptions) > 0:
                self.people_detection_publisher.publish(people_recognition)       
        else:
            debug_img = results[0].plot()[:, :, ::-1]          
            self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = rospy.get_param("~publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("~publishers/people_detection/topic", "/butia_vision/br/people_detection")
        self.people_detection_qs = rospy.get_param("~publishers/people_detection/queue_size", 1)

        self.threshold = rospy.get_param("~threshold", 0.5)


        self.all_classes = list(rospy.get_param("~all_classes", []))
        self.classes_by_category = dict(rospy.get_param("~classes_by_category", {}))

        r = rospkg.RosPack()
        r.list()
        self.model_file = r.get_path("butia_recognition") + "/weigths/" + rospy.get_param("~model_file", "yolov8n.pt")

        max_sizes = rospy.get_param("~max_sizes", [[0.05, 0.05, 0.05]])

        if len(max_sizes) == 1:
            max_sizes = [max_sizes[0] for _ in self.all_classes]

        if len(max_sizes) != len(self.all_classes):
            rospy.logerr('Wrong definition of max_sizes in yaml of recognition node.')
            assert False
        
        self.max_sizes = dict([(self.all_classes[i], max_sizes[i]) for i in range(len(max_sizes))])

        self.set_class_service = rospy.get_param("~servers/set_class", "/butia_vision/br/object_recognition/set_class")

        super().readParameters()

if __name__ == '__main__':
    rospy.init_node('yolov8_recognition_node', anonymous = True)

    yolo = YoloV8Recognition()

    rospy.spin()
