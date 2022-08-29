#!/usr/bin/env python3

from .deep_sort import nn_matching
from .deep_sort.tracker import Tracker 
from .deep_sort import preprocessing as prep
from .deep_sort.detection import Detection
from .deep_sort import generate_detections

from butia_vision_msgs.msg import Recognitions2D, Description2D

import cv2
import os

import numpy as np

class PeopleTracking():
    
    def __init__(self, model_path, matching_threshold = .5, max_iou_distance = 0.7, max_age=60, n_init=5):
        self.encoder = generate_detections.create_box_encoder(os.path.abspath(model_path))
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", matching_threshold)
        self.tracker = Tracker(self.metric,max_iou_distance,max_age,n_init)
        self.detections = []
        self.trackingPerson = None

    def setFrame(self, image_header, header, frame_id, frame):
        self.image_header = image_header
        self.header = header
        self.frame_id = frame_id
        self.frame = frame
        
    def track(self, descriptions):
        
        if len(descriptions) == 0:
            self.tracker.predict()
            print("No detections")
            return self.tracker, []

        detections = np.array([(int(d.bbox.center.x-d.bbox.size_x/2), int(d.bbox.center.y-d.bbox.size_y/2), d.bbox.size_x, d.bbox.size_y) for d in descriptions])

        out_scores = [d.score for d in descriptions]

        features = self.encoder(self.frame, detections)

        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(detections, out_scores, features)]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker, dets

    def startTrack(self):
        bigger_bbox = None
        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                if bigger_bbox is None:
                    self.trackingPerson = track
                    bigger_bbox = track.to_tlwh()[2]*track.to_tlwh()[3]
                else:
                    if bigger_bbox < track.to_tlwh()[2]*track.to_tlwh()[3]:
                        bigger_bbox = track.to_tlwh()[2]*track.to_tlwh()[3]
                        self.trackingPerson = track
    
    def stopTrack(self):
        self.trackingPerson = None

    def findPerson(self):
        if not self.trackingPerson.is_confirmed():
            for track in self.tracker.tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    distance = self.metric._metric(self.trackingPerson.features, track.features)
                    if distance < 0.7:
                        self.trackingPerson = track
                        break
                    



