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
    
    def __init__(self, model_path):
        self.encoder = generate_detections.create_box_encoder(os.path.abspath(model_path))
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", .5, 100)
        self.tracker = Tracker(self.metric)
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

    def startTrack():
        Bigbb = None
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                if Bigbb is not None:
                    self.trackingPerson = track
                    Bigbb = track.to_tlwh()[2]*track.to_tlwh()[3]
                else:
                    if Bigbb < track.to_tlwh()[2]*track.to_tlwh()[3]:
                        Bigbb = track.to_tlwh()[2]*track.to_tlwh()[3]
                        self.trackingPerson = track
    
    def stopTrack():
        self.trackingPerson = None



