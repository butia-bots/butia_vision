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
from butia_vision_msgs.srv import SetClass, SetClassRequest, SetClassResponse
from butia_vision_msgs.srv import VisualQuestionAnswering, VisualQuestionAnsweringRequest, VisualQuestionAnsweringResponse
import supervision as sv
from groundingdino.util.inference import Model
from segment_anything_hq import SamPredictor, sam_model_registry
from ram.models import ram
from ram import inference_ram
from ram import get_transform as get_transform_ram
from PIL import Image as PILImage
from transformers import pipeline
import gc

torch.set_num_threads(1)

class GroundedSAMRecognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)

        self.current_img = None
        self.detect_best_only = True
        self.readParameters()

        self.loadModel()
        self.initRosComm()

    def initRosComm(self):
        self.debug_publisher = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.object_recognition_publisher = rospy.Publisher(self.object_recognition_topic, Recognitions2D, queue_size=self.object_recognition_qs)
        self.set_class = rospy.Service(self.set_class_service, SetClass, self.handleSetClass)
        self.visual_question_answering = rospy.Service(self.visual_question_answering_service, VisualQuestionAnswering, self.handleVisualQuestionAnswering)
        super().initRosComm(callbacks_obj=self)

    def serverStart(self, req):
        self.loadModel()
        return super().serverStart(req)

    def serverStop(self, req):
        self.unLoadModel()
        return super().serverStop(req)

    def handleSetClass(self, req):
        self.classes = [req.class_name,]
        self.box_threshold = req.confidence
        self.detect_best_only = req.detect_best_only
        return SetClassResponse()

    def handleVisualQuestionAnswering(self, req):
        question = req.question
        image = PILImage.fromarray(self.current_img.copy())
        result = self.vqa_model(question=question, image=image, top_k=1)[0]
        answer = str(result['answer'])
        confidence = result['score']
        return VisualQuestionAnsweringResponse(answer=answer, confidence=confidence)

    def loadModel(self):
        self.dino_model = Model(model_config_path=f"{self.pkg_path}/config/grounding_dino_network_config/{self.dino_config}", model_checkpoint_path=f"{self.pkg_path}/config/grounding_dino_network_config/{self.dino_checkpoint}")
        print('Done loading GroundingDINO model!')
        if self.use_sam:
            sam = sam_model_registry[self.sam_model_type](checkpoint=f"{self.pkg_path}/config/sam_network_config/{self.sam_checkpoint}")
            self.sam_model = SamPredictor(sam)
            print('Done loading SAM model!')
        if self.use_ram:
            self.ram_model = ram(pretrained=f"{self.pkg_path}/config/ram_network_config/ram_swin_large_14m_no_optimizer.pth", vit="swin_l", image_size=384)
            self.ram_model.eval()
            self.ram_model = self.ram_model.to('cuda')
            self.ram_transform = get_transform_ram(image_size=384)
        if self.use_vqa:
            self.vqa_model = pipeline(model="dandelin/vilt-b32-finetuned-vqa")

    def unLoadModel(self):
        del self.dino_model
        if self.use_sam:
            del self.sam_model
        if self.use_ram:
            del self.ram_model
            del self.ram_transform
        if self.use_vqa:
            del self.vqa_model
        gc.collect()
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
            self.current_img = cv_img.copy()

            if self.use_ram:
                ram_img = self.ram_transform(PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to('cuda')
                ram_results = inference_ram(ram_img, self.ram_model)
                class_list = [class_name.strip() for class_name in ram_results[0].split('|')]
            else:
                class_list = self.classes

            print(class_list)
            results = self.dino_model.predict_with_classes(image=cv_img, classes=class_list, box_threshold=self.box_threshold, text_threshold=self.text_threshold)
            results = results.with_nms(threshold=self.nms_threshold, class_agnostic=self.class_agnostic_nms)
            if len(results.class_id) > 0:
                if self.detect_best_only:
                    results = results[results.confidence >= max(results.confidence)][:1]
                self.sam_model.set_image(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            box_annotator = sv.BoxAnnotator()
            print(results.class_id)
            labels = []
            for idx, conf in zip(results.class_id, results.confidence):
                if idx is not None:
                    labels.append(f"{class_list[idx]} {conf:.2f}")
                else:
                    labels.append(f'unknown {conf:.2f}')
            debug_img = box_annotator.annotate(scene=cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB), detections=results, labels=labels)
            mask_annotator = sv.MaskAnnotator()
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
            mask_arr = []
            for i in range(len(results.class_id)):
                if results.class_id[i] == None:
                    continue
                class_id = int(results.class_id[i])

                if class_id >= len(class_list):
                    continue

                label_class = class_list[class_id]

                if not self.use_ram:
                    max_size = [0., 0., 0.]
                    if class_id < len(self.max_sizes):
                        max_size = self.max_sizes[class_id]
                else:
                    max_size = [10., 10., 10.]

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
                if self.use_sam:
                    box_prompt = results.xyxy[i].astype(int)
                    masks, scores, logits = self.sam_model.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box_prompt,
                        multimask_output=False,
                        hq_token_only=self.sam_hq_token_only
                    )
                    mask_image = np.zeros(shape=(*cv_img.shape[:2],), dtype=bool)
                    for mask in masks:
                        mask_image = mask_image + mask.reshape(*mask_image.shape)
                    mask_msg = ros_numpy.msgify(Image, (mask_image*255).astype(np.uint8), 'mono8')
                    description.mask = mask_msg
                    mask_arr.append(mask_image)
                
                results.mask = np.asarray(mask_arr)

                index = None
                j = 0
                for value in self.classes_by_category.values():
                    if label_class in value:
                        index = j
                    j += 1
                description.label = class_list[index] + '/' + label_class if index is not None else label_class

                objects_recognition.descriptions.append(description)

                description_header.seq += 1
            
            debug_img = mask_annotator.annotate(debug_img, detections=results)
            
            self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))

            if len(objects_recognition.descriptions) > 0:
                self.object_recognition_publisher.publish(objects_recognition)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = rospy.get_param("~publishers/object_recognition/queue_size", 1)

        #tabletop recognition test: 8/14
        self.dino_checkpoint = rospy.get_param("~dino_checkpoint", "groundingdino_swint_ogc.pth")
        self.dino_config = rospy.get_param("~dino_config", "GroundingDINO_SwinT_OGC.py")
        
        self.class_agnostic_nms = rospy.get_param("~class_agnostic_nms", True)
        self.nms_threshold = rospy.get_param("~nms_threshold", 0.5)
        self.text_threshold = rospy.get_param("~text_threshold", 0.25)
        self.box_threshold = rospy.get_param("~box_threshold", 0.35)

        self.use_sam = rospy.get_param("~use_sam", True)
        self.sam_checkpoint = rospy.get_param("~sam_checkpoint", "sam_hq_vit_tiny.pth")
        self.sam_model_type = rospy.get_param("~sam_model_type", "vit_tiny")
        self.sam_hq_token_only = rospy.get_param("~sam_hq_token_only", False)

        self.use_ram = rospy.get_param("~use_ram", True)

        self.use_vqa = rospy.get_param("~use_vqa", True)
        self.visual_question_answering_service = rospy.get_param("~servers/visual_question_answering/service", "/butia_vision/br/object_recognition/visual_question_answering")

        self.classes_by_category = dict(rospy.get_param("~classes_by_category", {}))
        self.classes = rospy.get_param("~classes", [])
        self.set_class_service = rospy.get_param("~servers/set_class/service", "/butia_vision/br/object_recognition/set_class")

        self.max_sizes = list((rospy.get_param("~max_sizes", [])))

        super().readParameters()

if __name__ == '__main__':
    rospy.init_node('grounded_sam_recognition_node', anonymous = True)

    grounded_sam = GroundedSAMRecognition()

    rospy.spin()
