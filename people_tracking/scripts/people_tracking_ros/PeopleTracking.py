#!/usr/bin/env python

from deep_sort import nn_matching
from deep_sort.tracker import Tracker 
from deep_sort import preprocessing as prep
from deep_sort import visualization
from deep_sort.detection import Detection
from deep_sort import generate_detections

from butia_vision_msgs.msg import Recognitions, Description, BoundingBox
from std_msgs.msg import Header
from sensor_msgs.msg import Image

import cv2
import os

import numpy as np

class PeopleTracking():
    
    def __init__(self, model_path, matching_threshold = .5, max_iou_distance = 0.7, max_age=60, n_init=5):
        print(matching_threshold)
        print(max_iou_distance)
        print(max_age)
        print(n_init)
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
        
        if descriptions == []:
            self.tracker.predict()
            print("No detections")
            trackers = self.tracker.tracks 
            return trackers, None

        detections = np.array([(description.bounding_box.minX, description.bounding_box.minY, description.bounding_box.width, description.bounding_box.height) for description in descriptions])

        out_scores = [description.probability for description in descriptions]

        features = self.encoder(self.frame, detections)

        dets = [Detection(bbox, score, feature) for bbox, score, feature in zip(detections, out_scores, features)]

        outboxes = np.array([d.tlwh for d in dets])
        outscores = np.array([d.confidence for d in dets])

        indices = prep.non_max_suppression(outboxes, 0.5, outscores)

        dets = [dets[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker,dets

    def startTrack(self):
        Bigbb = None
        for track in self.tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                if Bigbb is None:
                    self.trackingPerson = track
                    Bigbb = track.to_tlwh()[2]*track.to_tlwh()[3]
                else:
                    if Bigbb < track.to_tlwh()[2]*track.to_tlwh()[3]:
                        Bigbb = track.to_tlwh()[2]*track.to_tlwh()[3]
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
                    



