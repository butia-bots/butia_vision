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
        self.recognitionPub = rospy.Publisher(self.recognition_topic, Recognitions2D, queue_size=self.recognition_qs)
        
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
        self.loadTrackerModel()
        self.trackID = -1
        self.lastTrack = 0
        self.tracking = True
        return EmptyResponse()
    
    def stopTracking(self, req):
        self.tracking = False
        self.unLoadTrackerModel()
        return EmptyResponse()
    
    def loadModel(self):
        self.model = YOLO(self.model_file)
        if self.tracking:
            self.loadTrackerModel()
    
    def loadTrackerModel(self):
        self.tracker = DeepOCSORT(
            model_weights=Path(self.reid_model_file),
            device="cuda:0",
            fp16=True,
            det_thresh=self.reid_threshold,
            max_age=self.max_age,
            iou_threshold=self.iou_threshold)
        return
    
    def unLoadTrackerModel(self):
        del self.tracker
        return
            
    
    def unLoadModel(self):
        del self.model
        self.unLoadTrackingModel
        return
    
    @ifState
    def callback(self, *args):
        img = None
        points = None
        
        data = self.sourceDataFromArgs(args)
        
        img = data["image_rgb"]
        img_depth = data["image_depth"]
        camera_info = data["camera_info"]

        IMG_HEIGHT = img.height
        IMG_WIDTH  = img.width

        HEADER = img.header

        recognition = Recognitions2D()
        recognition.image_rgb = img
        recognition.points = points
        recognition.image_depth = img_depth
        recognition.camera_info = camera_info
        recognition.header = HEADER
        recognition.descriptions = []
        # rospy.logwarn(recognition)
       
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
        # descriptions = []
        for box in bboxs:
            description = Description2D()
            description.header = HEADER

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
            

            # if description.label == "person":
            #     # peopleDetection.descriptions.append(description)
            #     if self.tracking:
            #         people_ids.append(ID)
            # else:
            #     objRecognition.descriptions.append(description)
            
            box_label = ""
            previus_dist = float("inf")
            center = (IMG_WIDTH/2,IMG_HEIGHT/2)
            if self.tracking:
                description.global_id = ID
                if description.label == "person":
                    people_ids.append(ID)
                
                box_label = f"ID:{ID} "
                dist = np.sqrt(np.power(description.bbox.center.x-center[0],2)+np.power(description.bbox.center.y-center[1],2))
                if ID == self.trackID or \
                    self.trackID == -1 or \
                    (perf_counter() - self.lastTrack >= self.max_age and \
                    (tracked_box == None or\
                    dist < previus_dist)):

                    self.trackID = ID
                    previus_dist = dist
                    tracked_box = description
            recognition.descriptions.append(description)
                    
            cv.rectangle(debug_img,(int(X1),int(Y1)), (int(X2),int(Y2)),(0,0,255),thickness=2)
            cv.putText(debug_img,f"{box_label}{self.model.names[clss]}:{score:.2f}", (int(X1), int(Y1)), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),thickness=2)
        
        if tracked_box != None:
            tracked_description = deepcopy(tracked_box)
            tracked_description.type = Description2D.TRACKING
            recognition.descriptions.append(tracked_box)
            self.lastTrack = now
            cv.rectangle(debug_img,(int(tracked_box.bbox.center.x-tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.y-tracked_box.bbox.size_y/2)),\
                                    (int(tracked_box.bbox.center.x+tracked_box.bbox.size_x/2),\
                                    int(tracked_box.bbox.center.y+tracked_box.bbox.size_y/2)),(255,0,0),thickness=2)
            
        
        poses = results[0].keypoints.data.cpu().numpy()
        for id, pose in zip(people_ids, poses):
            description = Description2D()
            description.header = HEADER
            description.type = Description2D.POSE
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
            recognition.descriptions.append(description)

        debug_msg = ros_numpy.msgify(Image, debug_img, encoding='bgr8')
        debug_msg.header = HEADER
        self.debugPub.publish(debug_msg)
        
        if len(recognition.descriptions) > 0:
            # rospy.logwarn("OI"*1000)
            # recognition.descriptions = descriptions
            self.recognitionPub.publish(recognition)
        
    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic","/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/pose_recognition/queue_size",1)

        self.recognition_topic = rospy.get_param("~publishers/recognition/topic", "/butia_vision/br/recognition")
        self.recognition_qs = rospy.get_param("~publishers/recognition/queue_size", 1)

        self.start_tracking_topic = rospy.get_param("~services/tracking/start","/butia_vision/pt/start")
        self.stop_tracking_topic = rospy.get_param("~services/tracking/stop","/butia_vision/pt/stop")

        self.threshold = rospy.get_param("~debug_kpt_threshold", 0.5)

        self.model_file = rospy.get_param("~model_file","yolov8n-pose")
        self.reid_model_file = rospy.get_param("~reid_model_file","osnet_x0_25_msmt17.pt")

        self.reid_threshold = rospy.get_param("~tracking/thresholds/reid_threshold", 0.5)
        self.iou_threshold = rospy.get_param('~tracking/thresholds/iou_threshold',0.5)
        self.max_age = rospy.get_param("~tracking/thresholds/max_age",60)
        self.tracking_on_init = rospy.get_param("~tracking/start_on_init", False)

        self.tracking = False

        super().readParameters()
    
   
if __name__ == "__main__":
    rospy.init_node("yolo_tracjer_recognition_node", anonymous = True)
    yolo = YoloTrackerRecognition()
    rospy.spin()