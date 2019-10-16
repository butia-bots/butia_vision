#!/usr/bin/env python

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection

from butia_vision_msgs.msg import Recognition, Description, BoundingBox
from std_msgs.msg import Header
from sensor_msgs.msg import Image

import FeatureGenerator

import numpy as np

class PeopleTracking():
    
    def __init__(self, feature_generator):
        self.feature_generator = feature_generator
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", .5, 100)
        self.tracker = Tracker(self.metric)
        self.detections = []

    def setFrame(image_header, header, frame_id, frame)
        self.image_header = image_header
        self.header = header
        self.frame_id = frame_id
        self.frame = frame

    def generateDetections(descriptions, features):
        for i in range(len(descriptions)):
            bbox = (description[i].minX, description[i].minY, description[i].width, description[i].height)   
            self.detections.append(Detection(bbox, description[i].probability, features[i]))
        
    def track():
        
        if self.detections == []:
            self.tracker.predict()
            print("No detections")
            trackers = self.tracker.tracks 
            return trackers
        
        outboxes = np.array([self.detections.tlwh for d in self.detections])
        outscores = np.array([self.detections.confidence for d in self.detections])

        indices = prep.non_max_suppression(outboxes, 0.8, outscores)

        dets = [self.detections[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)

        return self.tracker,dets

