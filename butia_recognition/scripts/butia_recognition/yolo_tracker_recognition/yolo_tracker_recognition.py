#!/usr/bin/env python

from butia_recognition import BaseRecognition, ifState

import rospy
import cv2 as cv
import numpy as np
import ros_numpy

from pathlib import Path
from time import perf_counter

from ultralytics import YOLO
from boxmot import DeepOCSORT

from sensor_msgs.msg import Image
from std_srvs.srv import EmptyResponse, Empty

from butia_vision_msgs.msg import Recognitions2D, Description2D, KeyPoint
from copy import deepcopy



class YoloTrackerRecognition(BaseRecognition):
    def __init__(self,state = True):
        super().__init__(state=state)
        self.readParameters()
        self.loadModel()
        self.initRosComm()
        if self.tracking_on_init:
            self.startTracking(None)
        rospy.loginfo("Yolo Tracker Recognition started")

    def initRosComm(self):
        self.debugPub = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.objRecognitionPub = rospy.Publisher(self.object_recognition_topic, Recognitions2D, queue_size=self.object_recognition_qs)
        self.peopleDetectionPub = rospy.Publisher(self.people_detection_topic, Recognitions2D, queue_size=self.people_detection_qs)
        self.poseRecognitionPub = rospy.Publisher(self.pose_recognition_topic, Recognitions2D, queue_size=self.pose_recognition_qs)
        self.trackingPub = rospy.Publisher(self.tracking_topic,Recognitions2D, queue_size=self.tracking_qs)

        self.trackingStartService = rospy.Service(self.start_tracking_topic, Empty, self.startTracking)
        self.trackingStopService = rospy.Service(self.stop_tracking_topic, Empty, self.stopTracking)
        super().initRosComm(callbacks_obj=self)

    
    def serverStart(self, req):
        self.loadModel()
        
        return super().serverStart(req)
    
    def serverStop(self, req):
        self.unLoadModel()
        return super().serverStop(req)
    
    def startTracking(self, req):
        self.tracking = True
        self.trackID = -1
        self.lastTrack = 0
        return
    
    def stopTracking(self, req):
        self.tracking = False
        return
    
    def loadModel(self):
        self.model = YOLO(self.model_file)
        self.tracker = DeepOCSORT(
            model_weights=Path(self.reid_model_file),
            device="cuda:0",
            fp16=True,
            det_thresh=self.reid_threshold,
            max_age=self.max_age,
            iou_threshold=self.iou_threshold
        )
    
    def unLoadModel(self):
        del self.model
        del self.tracker
    
    @ifState
    def callback(self, *args):
        img = None
        points = None
        if len(args):
            img = args[0]
            points = args[1]
        
        IMG_HEIGHT = img.height
        IMG_WIDTH  = img.width
        
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

        trackRecognition = Recognitions2D()
        trackRecognition.image_rgb = img
        trackRecognition.points = points
        trackRecognition.header = points.header
       
        img = ros_numpy.numpify(img)        

        debug_img = deepcopy(img)
        # print(debug_img == img)     
        results = self.model(img)

        bboxs = results[0].boxes.data.cpu().numpy()
        if self.tracking:
            bboxs  = self.tracker.update(bboxs, img)
        people_ids = []

        tracked_box = None
        now = perf_counter()
        for box in bboxs:
            description = Description2D()
            description.header = points.header

            X1,Y1,X2,Y2 = box[:4]
            ID = int(box[4]) if self.tracking else None
            score = box[5] if self.tracking else box[4]
            clss = int(box[6] if self.tracking else box[5])

            description.bbox.center.x = (X1+X2)/2
            description.bbox.center.y = (Y1+Y2)/2
            description.bbox.size_x = X2-X1
            description.bbox.size_y = Y2-Y1
            description.type = Description2D.DETECTION
            description.label = self.model.names[clss]

            if self.model.names[clss] == "person":
                peopleDetection.descriptions.append(description)
                if self.tracking:
                    people_ids.append(ID)
            else:
                objRecognition.descriptions.append(description)
            
            box_label = ""
            previus_dist = float("inf")
            center = (IMG_WIDTH/2,IMG_HEIGHT/2)
            if self.tracking:
                description.global_id = ID
                box_label = f"ID:{ID} "
                # Esse IF ta todo errado
                dist = np.sqrt(np.power(description.bbox.center.x-center[0],2)+np.power(description.bbox.center.y-center[1],2))
                if ID == self.trackID or \
                    self.trackID == -1 or \
                    (perf_counter() - self.lastTrack >= self.max_age and \
                    (tracked_box == None or\
                    dist < previus_dist)):

                    self.trackID = ID
                    previus_dist = dist
                    tracked_box = description
                    
            cv.rectangle(debug_img,(int(X1),int(Y1)), (int(X2),int(Y2)),(0,0,255),thickness=2)
            cv.putText(debug_img,f"{box_label}{self.model.names[clss]}:{score:.2f}", (int(X1), int(Y1)), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),thickness=2)
        if tracked_box != None:
            trackRecognition.descriptions.append(tracked_box)
            self.lastTrack = now
            cv.rectangle(debug_img,(int(tracked_box.bbox.center.x-tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.y-tracked_box.bbox.size_y/2)),\
                                    (int(tracked_box.bbox.center.x+tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.y+tracked_box.bbox.size_y/2)),(255,0,0),thickness=2)
            
        
        poses = results[0].keypoints.data.cpu().numpy()
        for id, pose in zip(people_ids, poses):
            description = Description2D()
            description.header = points.header
            description.header = Description2D.POSE
            description.global_id = id
            # rospy.logwarn(pose)
            for idx, kpt in enumerate(pose):
                keypoint = KeyPoint()
                keypoint.x = kpt[0]
                keypoint.y = kpt[1]
                keypoint.id = idx
                keypoint.score = kpt[2]
                description.pose.append(keypoint)
                if kpt[2] >= self.threshold:
                    cv.circle(debug_img, (int(kpt[0]), int(kpt[1])),3,(0,255,0),thickness=-1)
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
        if len(trackRecognition.descriptions) != 0:
            self.trackingPub.publish(trackRecognition)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic","/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/pose_recognition/queue_size",1)

        self.object_recognition_topic = rospy.get_param("~publishers/object_recognition/topic", "/butia_vision/br/object_recognition")
        self.object_recognition_qs = rospy.get_param("~publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("~publishers/people_detection/topic", "/butia_vision/br/people_detection")
        self.people_detection_qs = rospy.get_param("~publishers/people_detection/queue_size", 1)

        self.pose_recognition_topic = rospy.get_param("~publishers/pose_recognition/topic","/butia_vision/br/pose_detection")
        self.pose_recognition_qs = rospy.get_param("~publishers/pose_recognition/queue_size",1)

        self.tracking_topic = rospy.get_param("~publishers/track/topic","/butia_vision/pt/people_tracking")
        self.tracking_qs = rospy.get_param("~publishers/track/queue_size",1)

        self.start_tracking_topic = rospy.get_param("~services/tracking/start","/butia_vision/pt/start")
        self.stop_tracking_topic = rospy.get_param("~services/tracking/stop","/butia_vision/pt/stop")

        self.threshold = rospy.get_param("~debug_kpt_threshold", 0.5)

        self.model_file = rospy.get_param("~model_file","yolov8n-pose")
        self.reid_model_file = rospy.get_param("~reid_model_file","osnet_x0_25_msmt17.pt")


        self.reid_threshold = rospy.get_param("~tracking/thresholds/reid_threshold", 0.5)
        self.iou_threshold = rospy.get_param('~tracking/thresholds/iou_threshold',0.5)
        self.max_age = rospy.get_param("~tracking/thresholds/max_age",60)
        self.tracking_on_init = rospy.get_param("~tracking/start_on_init", False)

        super().readParameters()
    
   
if __name__ == "__main__":
    rospy.init_node("yolo_tracjer_recognition_node", anonymous = True)
    yolo = YoloTrackerRecognition()
    rospy.spin()