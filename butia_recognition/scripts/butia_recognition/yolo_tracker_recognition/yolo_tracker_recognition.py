#!/usr/bin/env python

from butia_recognition import BaseRecognition, ifState

import rospy
import cv2 as cv
import ros_numpy

from pathlib import Path

from ultralytics import YOLO

from sensor_msgs.msg import Image

from butia_vision_msgs.msg import Recognitions2D, Description2D, KeyPoint
from copy import deepcopy


class YoloTrackerRecognition(BaseRecognition):
    def __init__(self,state = True):
        super().__init__(state=state)
        self.readParameters()
        self.loadModel()
        self.initRosComm()
        rospy.logwarn("Finished starting")

    def initRosComm(self):
        self.debugPub = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.objRecognitionPub = rospy.Publisher(self.object_recognition_topic, Recognitions2D, queue_size=self.object_recognition_qs)
        self.peopleDetectionPub = rospy.Publisher(self.people_detection_topic, Recognitions2D, queue_size=self.people_detection_qs)
        self.poseRecognitionPub = rospy.Publisher(self.pose_recognition_topic, Recognitions2D, queue_size=self.pose_recognition_qs)
        super().initRosComm(callbacks_obj=self)

    
    def serverStart(self, req):
        self.loadModel()
        return super().serverStart(req)
    
    def serverStop(self, req):
        self.unLoadModel()
        return super().serverStop(req)

    def loadModel(self):
        self.model = YOLO(self.model_file)
        
    
    def unLoadModel(self):
        del self.model
    
    @ifState
    def callback(self, *args):
        img = None
        points = None
        if len(args):
            img = args[0]
            print(img.encoding)
            points = args[1]
        
        objRecognition = Recognitions2D()
        objRecognition.image_rgb = img
        objRecognition.points = points
        objRecognition.header = points.header

        peopleDetection = Recognitions2D()
        peopleDetection.image_rgb
        peopleDetection.points = points
        peopleDetection.header = points.header

        poseRecognition = Recognitions2D()
        poseRecognition.image_rgb = img
        poseRecognition.points = points
        poseRecognition.header = points.header
       
        img = ros_numpy.numpify(img)        

        debug_img = deepcopy(img)
        # print(debug_img == img)     
        results = self.model(img)

        for box in results[0].boxes.data.cpu().numpy():
            description = Description2D()
            description.header = points.header

            cls = int(box[5])
            description.bbox.center.x = (box[0]+box[2])/2
            description.bbox.center.y = (box[1]+box[3])/2
            description.bbox.size_x = box[2]-box[0]
            description.bbox.size_y = box[3]-box[1]
            description.type = Description2D.DETECTION
            description.label = self.model.names[cls]
            if self.model.names[cls] == "person":
                peopleDetection.descriptions.append(description)
            else:
                objRecognition.descriptions.append(description)
            cv.rectangle(debug_img,(int(box[0]),int(box[1])), (int(box[2]),int(box[3])),(0,0,255),thickness=2)
            cv.putText(debug_img,f"{self.model.names[int(box[5])]}:{box[4]:.2f}", (int(box[0]), int(box[1])), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),thickness=2)
        
        for pose in results[0].keypoints.cpu().numpy():
            description = Description2D()
            description.header = points.header
            for idx, kpt in enumerate(pose):
                keypoint = KeyPoint()
                keypoint.x = kpt[0]
                keypoint.y = kpt[1]
                keypoint.id = idx
                keypoint.score = kpt[2]
                description.pose.append(keypoint)
                if kpt[2] >= self.threshold:
                    cv.circle(debug_img, (int(kpt[0]), int(kpt[1])),5,(0,255,0),thickness=-1)
            poseRecognition.descriptions.append(description)

        debug_msg = ros_numpy.msgify(Image, debug_img, encoding='bgr8')
        debug_msg.header = points.header
        self.debugPub.publish(debug_msg)
        if len(peopleDetection.descriptions) != 0:
            self.peopleDetectionPub.publish(peopleDetection)
        if len(objRecognition.descriptions) != 0:
            self.objRecognitionPub.publish(objRecognition)
        if len(poseRecognition.descriptions) != 0:
            self.poseRecognitionPub.publish(poseRecognition)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic","/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/pose_recognition/queue_size",1)

        self.object_recognition_topic = rospy.get_param("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = rospy.get_param("~publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("~publishers/people_detection/topic", "/butia_vision/br/people_detection")
        self.people_detection_qs = rospy.get_param("~publishers/people_detection/queue_size", 1)

        self.pose_recognition_topic = rospy.get_param("~publishers/pose_recognition/topic","/butia_vision/br/pose_detection")
        self.pose_recognition_qs = rospy.get_param("~publishers/pose_recognition/queue_size",1)

        self.threshold = rospy.get_param("~threshold", 0.5)

        self.model_file = rospy.get_param("~model_file","yolov8n-pose")

        # self.config_path = rospy.get_param("~tracker_config_path", "")

        super().readParameters()
    
    # def on_predict_start(self, predictor):
    #     predictor.trackers = []
    #     predictor.tracker_outputs = [None] * predictor.dataset.bs
    #     predictor.args.tracking_config = self.config_path
    #     for i  in range(predictor.dataset.bs):
    #         tracker = create_tracker(
    #             predictor.args.tracking_method,
    #             predictor.args.tracking_config,
    #             predictor.args.reid_model,
    #             predictor.device,
    #             predictor.args.half
    #         )
    #         predictor.trackers.append(tracker)


if __name__ == "__main__":
    rospy.init_node("yolo_tracjer_recognition_node", anonymous = True)
    yolo = YoloTrackerRecognition()
    rospy.spin()