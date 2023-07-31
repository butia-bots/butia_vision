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
from geometry_msgs.msg import Vector3
from butia_vision_msgs.msg import Description2D, Recognitions2D
import supervision as sv
from groundingdino.util.inference import Model

torch.set_num_threads(1)

class GroundedSAMRecognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)

        self.readParameters()

        self.loadModel()
        self.initRosComm()

    def initRosComm(self):
        self.debug_publisher = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.object_recognition_publisher = rospy.Publisher(self.object_recognition_topic, Recognitions2D, queue_size=self.object_recognition_qs)
        super().initRosComm(callbacks_obj=self)

    def serverStart(self, req):
        self.loadModel()
        return super().serverStart(req)

    def serverStop(self, req):
        self.unLoadModel()
        return super().serverStop(req)

    def loadModel(self):
        self.dino_model = Model(model_config_path=f"{self.pkg_path}/config/grounding_dino_network_config/{self.dino_config}", model_checkpoint_path=f"{self.pkg_path}/config/grounding_dino_network_config/{self.dino_checkpoint}")
        print('Done loading model!')

    def unLoadModel(self):
        del self.dino_model
        torch.cuda.empty_cache()

    @ifState
    def callback(self, *args):
        source_data = self.sourceDataFromArgs(args)

        if 'image_rgb' not in source_data:
            rospy.logwarn('Souce data has no image_rgb.')
            return None
        
        img = source_data['image_rgb']

        with torch.no_grad():

            rospy.loginfo('Image ID: ' + str(img.header.seq))

            cv_img = ros_numpy.numpify(img)

            results = self.dino_model.predict_with_classes(image=cv_img, classes=self.classes, box_threshold=self.box_threshold, text_threshold=self.text_threshold)
            results = results.with_nms(threshold=self.nms_threshold, class_agnostic=self.class_agnostic_nms)
            annotator = sv.BoxAnnotator()
            debug_img = annotator.annotate(scene=cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), detections=results, labels=[self.classes[idx] for idx in results.class_id])

            objects_recognition = Recognitions2D()
            h = Header()
            h.seq = self.seq
            self.seq += 1
            h.stamp = rospy.Time.now()
            objects_recognition.header = h

            #adding source data
            objects_recognition = BaseRecognition.addSourceData2Recognitions2D(source_data, objects_recognition)

            people_recognition = copy(objects_recognition)

            description_header = img.header
            description_header.seq = 0
            for i in range(len(results.class_id)):
                class_id = int(results.class_id[i])

                if class_id >= len(self.classes):
                    continue

                label_class = self.classes[class_id]

                max_size = [0., 0., 0.]
                if class_id < len(self.max_sizes):
                    max_size = self.max_sizes[class_id]

                description = Description2D()
                description.header = copy(description_header)
                description.type = Description2D.DETECTION
                description.id = description.header.seq
                description.score = results.confidence[i]
                description.max_size = Vector3(*max_size)
                x1, y1, x2, y2 = results.xyxy[i]
                size = int(x2 - x1), int(y2 - y1)
                description.bbox.center.x = int(x1) + int(size[0]/2)
                description.bbox.center.y = int(y1) + int(size[1]/2)
                description.bbox.size_x = size[0]
                description.bbox.size_y = size[1]

                index = None
                j = 0
                for value in self.classes_by_category.values():
                    if label_class in value:
                        index = j
                    j += 1
                description.label = self.classes[index] + '/' + label_class if index is not None else label_class

                objects_recognition.descriptions.append(description)

                description_header.seq += 1
            
            self.debug_publisher.publish(ros_numpy.msgify(Image, np.flip(debug_img, 2), 'rgb8'))

            if len(objects_recognition.descriptions) > 0:
                self.object_recognition_publisher.publish(objects_recognition)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = rospy.get_param("~publishers/object_recognition/queue_size", 1)

        self.dino_checkpoint = rospy.get_param("~dino_checkpoint", "groundingdino_swint_ogc.pth")
        self.dino_config = rospy.get_param("~dino_config", "GroundingDINO_SwinT_OGC.py")
        self.class_agnostic_nms = rospy.get_param("~class_agnostic_nms", True)
        self.nms_threshold = rospy.get_param("~nms_threshold", 0.5)
        self.text_threshold = rospy.get_param("~text_threshold", 0.25)
        self.box_threshold = rospy.get_param("~box_threshold", 0.35)


        self.classes_by_category = dict(rospy.get_param("~classes_by_category", {}))
        self.classes = rospy.get_param("~classes", [])

        self.max_sizes = list((rospy.get_param("~max_sizes", [])))

        super().readParameters()

if __name__ == '__main__':
    rospy.init_node('grounded_sam_recognition_node', anonymous = True)

    grounded_sam = GroundedSAMRecognition()

    rospy.spin()
