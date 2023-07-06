#!/usr/bin/env python

from butia_recognition import BaseRecognition, ifState

import rospy
import cv2 as cv
import numpy as np
import ros_numpy
import rospkg

from pathlib import Path
from time import perf_counter

from ultralytics import YOLO

from sensor_msgs.msg import Image
from std_srvs.srv import EmptyResponse, Empty

from butia_vision_msgs.msg import Recognitions2D, Description2D, KeyPoint2D, Recognitions3D
from copy import deepcopy

import gc
import torch

DeepOCSORT = None

class YoloTrackerRecognition(BaseRecognition):
    def __init__(self,state = True):
        super().__init__(state=state)
        self.readParameters()
        self.loadModel()
        self.initRosComm()
        if self.use_boxmot:
            import boxmot
            global DeepOCSORT
            DeepOCSORT = boxmot.DeepOCSORT
        if self.tracking_on_init:
            self.startTracking(None)

        rospy.loginfo("Yolo Tracker Recognition Node started")

    def initRosComm(self):
        self.debugPub = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.recognitionPub = rospy.Publisher(self.recognition_topic, Recognitions2D, queue_size=self.recognition_qs)
        self.trackingPub = rospy.Publisher(self.tracking_topic, Recognitions2D, queue_size=self.tracking_qs)
        
        self.trackingStartService = rospy.Service(self.start_tracking_topic, Empty, self.startTracking)
        self.trackingStopService = rospy.Service(self.stop_tracking_topic, Empty, self.stopTracking)
        super().initRosComm(callbacks_obj=self)

    
    def serverStart(self, req):
        self.loadModel()
        return super().serverStart(req)
    
    def serverStop(self, req):
        self.unLoadModel()
        self.stopTracking(None)
        return super().serverStop(req)
    
    def startTracking(self, req):
        self.loadTrackerModel()
        self.trackID = -2
        self.lastTrack = -self.max_time
        self.tracking = True
        rospy.loginfo("Tracking started!!!")
        return EmptyResponse()
    
    def stopTracking(self, req):
        self.tracking = False
        self.unLoadTrackerModel()
        rospy.loginfo("Tracking stoped!!!")
        return EmptyResponse()
    
    def loadModel(self):
        self.model = YOLO(self.model_file)
        if self.tracking:
            self.loadTrackerModel()
    
    def loadTrackerModel(self):
        if self.use_boxmot:
            self.tracker = DeepOCSORT(
                model_weights=Path(self.reid_model_file),
                device="cuda:0",
                fp16=True,
                det_thresh=self.det_threshold,
                max_age=self.max_age,
                iou_threshold=self.iou_threshold)
        return
    
    def unLoadTrackerModel(self):
        if self.use_boxmot:
            del self.tracker
            torch.cuda.empty_cache()
            self.tracker = None
        return
            
    
    def unLoadModel(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None
        return
    
    @ifState
    def callback(self, *args):

        tracking = self.tracking
                
        data = self.sourceDataFromArgs(args)
        
        img = data["image_rgb"]
        img_depth = data["image_depth"]
        camera_info = data["camera_info"]
        HEADER = img.header

        recognition = Recognitions2D()
        recognition.image_rgb = img
        recognition.image_depth = img_depth
        recognition.camera_info = camera_info
        recognition.header = HEADER
        recognition.descriptions = []
        img = ros_numpy.numpify(img)        

        debug_img = deepcopy(img)
        results = None
        bboxs   = None

        if tracking and self.use_boxmot:
            results = list(self.model.predict(img, verbose=False, stream=True))
            bboxs  = self.tracker.update(results[0].boxes.data.cpu().numpy(),img)
        elif tracking:
            results = list(self.model.track(img, persist=True,
                                        conf=self.det_threshold,
                                        iou=self.iou_threshold,
                                        device="cuda:0",
                                        tracker=self.tracker_cfg_file,
                                        verbose=True, stream=True))
            bboxs = results[0].boxes.data.cpu().numpy()
        else:
            results = list(self.model.predict(img, verbose=False, stream=True))
            bboxs = results[0].boxes.data.cpu().numpy()

        people_ids = []

        tracked_box = None
        now = perf_counter()
        # descriptions = []
        for i, box in enumerate(bboxs):
            description = Description2D()
            description.header = HEADER

            X1,Y1,X2,Y2 = box[:4]
            ID = int(box[4]) if self.tracking and len(box) == 7 else -1
            score = box[-2]
            clss = int(box[-1])

            description.bbox.center.x = (X1+X2)/2
            description.bbox.center.y = (Y1+Y2)/2
            description.bbox.size_x = X2-X1
            description.bbox.size_y = Y2-Y1
            description.label = self.model.names[clss]
            description.type = Description2D.DETECTION
            description.score = score
            description.id = i

            box_label = ""
            previus_size = float("-inf")
            if tracking:
                description.global_id = ID
                if description.label == "person":
                    people_ids.append(ID)                 
                
                box_label = f"ID:{ID} "
                size = description.bbox.size_x * description.bbox.size_y
                if ID == self.trackID or \
                    self.trackID == -1 or \
                    (perf_counter() - self.lastTrack >= self.max_time and \
                    (tracked_box == None or\
                    size > previus_size)):

                    self.trackID = ID
                    previus_size = size
                    tracked_box = description
                    self.lastTrack = perf_counter()

                if ID == self.trackID or self.trackID >= -1:
                    self.trackID = ID
                    previus_size = size
                    tracked_box = description
                    self.lastTrack = perf_counter()
                elif perf_counter() - self.lastTrack >= self.max_time:
                    if tracked_box == None or size > previus_size:
                        self.trackID = ID
                        previus_size = size
                        tracked_box = description
                        self.lastTrack = perf_counter()

            recognition.descriptions.append(description)
                    
            cv.rectangle(debug_img,(int(X1),int(Y1)), (int(X2),int(Y2)),(0,0,255),thickness=2)
            cv.putText(debug_img,f"{box_label}{self.model.names[clss]}:{score:.2f}", (int(X1), int(Y1)), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),thickness=2)
        
        track_recognition = Recognitions2D()
        track_recognition.header = HEADER
        tracked_description : Description2D = deepcopy(tracked_box)
        if tracked_box != None:
            track_recognition.header = recognition.header
            track_recognition.image_rgb = recognition.image_rgb
            track_recognition.image_depth = recognition.image_depth
            track_recognition.camera_info = recognition.camera_info
            tracked_description.type = Description2D.DETECTION
            # recognition.descriptions.append(tracked_box)
            self.lastTrack = now
            cv.rectangle(debug_img,(int(tracked_box.bbox.center.x-tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.y-tracked_box.bbox.size_y/2)),\
                                    (int(tracked_box.bbox.center.x+tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.y+tracked_box.bbox.size_y/2)),(255,0,0),thickness=2)
            
        
        if results[0].keypoints != None:
            poses = results[0].keypoints.data.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            poses_idx = scores > self.det_threshold
            poses = poses[poses_idx]
            counter = 0
            # if not tracking or len(people_ids) == len(poses):
            desc : Description2D
            for desc in recognition.descriptions:
                if desc.label == "person" and desc.score >= self.det_threshold:
                    desc.type = Description2D.POSE
                    # rospy.logwarn(desc.score)
                    for idx, kpt in enumerate(poses[counter]):
                        keypoint = KeyPoint2D()
                        keypoint.x = kpt[0]
                        keypoint.y = kpt[1]
                        keypoint.id = idx
                        keypoint.score = kpt[2]
                        desc.pose.append(keypoint)
                        if kpt[2] >= self.threshold:
                            cv.circle(debug_img, (int(kpt[0]), int(kpt[1])),3,(0,255,0),thickness=-1)
                        if tracking:
                            desc.global_id = people_ids[counter]
                    if tracked_box != None and tracked_description.global_id == desc.global_id:
                        desc.header = HEADER
                        tracked_description = desc
                        for kpt in desc.pose:
                            if kpt.score >= self.threshold:
                                cv.circle(debug_img, (int(kpt.x), int(kpt.y)),3,(0,255,255),thickness=-1)
                    counter +=1

        track_recognition.descriptions.append(tracked_description)
        debug_msg = ros_numpy.msgify(Image, debug_img, encoding='bgr8')
        debug_msg.header = HEADER
        self.debugPub.publish(debug_msg)
        
        if len(recognition.descriptions) > 0:
            self.recognitionPub.publish(recognition)
        if tracked_box != None and len(track_recognition.descriptions) > 0:
            self.trackingPub.publish(track_recognition)
        
    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic","/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/pose_recognition/queue_size",1)

        self.recognition_topic = rospy.get_param("~publishers/recognition/topic", "/butia_vision/br/recognition")
        self.recognition_qs = rospy.get_param("~publishers/recognition/queue_size", 1)

        self.start_tracking_topic = rospy.get_param("~services/tracking/start","/butia_vision/pt/start")
        self.stop_tracking_topic = rospy.get_param("~services/tracking/stop","/butia_vision/pt/stop")
        
        self.tracking_topic = rospy.get_param("~publishers/tracking/topic", "pub/tracking2d")
        self.tracking_qs = rospy.get_param("~publishers/tracking/queue_size", 1)

        self.threshold = rospy.get_param("~debug_kpt_threshold", 0.5)

        r = rospkg.RosPack()

        r.list()
        self.model_file = r.get_path("butia_recognition") + "/weigths/" + rospy.get_param("~model_file","yolov8n-pose")
        self.reid_model_file = rospy.get_param("~tracking/model_file","osnet_x0_25_msmt17.pt")

        self.det_threshold = rospy.get_param("~tracking/thresholds/det_threshold", 0.5)
        self.reid_threshold = rospy.get_param("~tracking/thresholds/reid_threshold", 0.3)
        self.iou_threshold = rospy.get_param('~tracking/thresholds/iou_threshold',0.5)
        self.max_time = rospy.get_param("~tracking/thresholds/max_time",60)
        self.max_age = rospy.get_param("~tracking/thresholds/max_age",5)
        self.tracking_on_init = rospy.get_param("~tracking/start_on_init", False)
        self.use_boxmot = rospy.get_param("~tracking/use_boxmot", False)

        self.tracker_cfg_file = r.get_path("butia_recognition") + "/" + rospy.get_param("~tracker-file","")

        self.tracking = False

        super().readParameters()
    
   
if __name__ == "__main__":
    rospy.init_node("yolo_tracker_recognition_node", anonymous = True)
    yolo = YoloTrackerRecognition()
    rospy.spin()