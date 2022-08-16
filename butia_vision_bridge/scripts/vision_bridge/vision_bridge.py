#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

import message_filters


class VisionBridge:
    SOURCE_MESSAGE_TYPES = {
        'image_rgb': Image,
        'camera_info_rgb': CameraInfo,
        'image_depth': Image,
        'camera_info_depth': CameraInfo,
        'point_cloud': PointCloud2
    }

    def __init__(self):
        self.readParameters()

        self.createSubscribers()
    
    def processData(self, data: Image):
        pass
    
    def processData(self, data: CameraInfo):
        pass
    
    def processData(self, data: PointCloud2):
        pass

    def callback(self, *args):
        pass
    
    def createSubscribers(self):
        self.subscribers = []
        for source in self.sync_sources:
            self.subscribers.append(message_filters.Subscriber(source, VisionBridge.SOURCE_MESSAGE_TYPES[source]))
        if self.use_exact_sync:
            self.time_synchronizer = message_filters.TimeSynchronizer(self.subscribers, self.queue_size)
        else:
            self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(self.subscribers, self.queue_size)
        self.time_synchronizer.registerCallback(self.callback)

    def readParameters(self):
        self.sync_sources = list(rospy.get_param('sync_sources', ['image_rgb', 'point_cloud']))
        assert all([source in self.sync_sources for source in self.sync_sources])
        self.queue_size = rospy.get_param('queue_size', 1)
        self.use_exact_sync = rospy.get_param('exact_sync', False)
        size = tuple(rospy.get_param('size', [640, 480]))
        assert len(size) == 2
        self.width, self.height = size