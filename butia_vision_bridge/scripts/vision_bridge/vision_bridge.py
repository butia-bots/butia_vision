#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

import message_filters

from multipledispatch import dispatch

class VisionBridge:
    SOURCE_MESSAGE_TYPES = {
        'image_rgb': Image,
        'camera_info_rgb': CameraInfo,
        'image_depth': Image,
        'camera_info_depth': CameraInfo,
        'points': PointCloud2
    }

    def __init__(self):
        self.readParameters()

        self.createSubscribersAndPublishers()
    
    @dispatch(Image)
    def processData(self, data: Image):
        image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        image = cv2.resize(image, (self.height, self.width), cv2.INTER_LINEAR)

        data.data = image.flatten().tolist()
        data.width = self.width
        data.height = self.height
        return data
    
    @dispatch(CameraInfo)
    def processData(self, data: CameraInfo):
        scale_x = self.width/data.width
        scale_y = self.height/data.height

        data.K[0] *= scale_x
        data.K[2] *= scale_x
        data.K[4] *= scale_y
        data.K[5] *= scale_y

        data.width = self.width
        data.height = self.height

        return data
    
    @dispatch(PointCloud2)
    def processData(self, data: PointCloud2):
        pc = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width, -1)
        print(pc)
        # since point cloud is organized, I am using opencv to resize it
        pc = cv2.resize(pc, (self.height, self.width), cv2.INTER_LINEAR)
        #print(pc)

        data.data = pc.flatten().tolist()
        data.width = self.width
        data.height = self.height
        return data

    def callback(self, *args):
        publish_dict = {}
        for i, data in enumerate(args):
            source = self.sync_sources[i]
            publish_dict[source] = self.processData(data)
        self.publish(publish_dict)

    def publish(self, data_dict):
        for source, data in data_dict.items():
            self.publishers[source].publish(data)
    
    def createSubscribersAndPublishers(self):
        self.subscribers = []
        self.publishers = {}
        for source in self.sync_sources:
            self.subscribers.append(message_filters.Subscriber('sub/' + source, VisionBridge.SOURCE_MESSAGE_TYPES[source]))
            self.publishers[source] = rospy.Publisher('pub/' + source, VisionBridge.SOURCE_MESSAGE_TYPES[source], queue_size=self.queue_size)
        if self.use_exact_sync:
            self.time_synchronizer = message_filters.TimeSynchronizer(self.subscribers, self.queue_size)
        else:
            self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(self.subscribers, self.queue_size, self.slop)
        self.time_synchronizer.registerCallback(self.callback)

    def readParameters(self):
        self.sync_sources = list(rospy.get_param('~sync_sources', ['image_rgb', 'points']))
        assert all([source in self.sync_sources for source in self.sync_sources])
        self.queue_size = rospy.get_param('~queue_size', 1)
        self.use_exact_sync = rospy.get_param('~exact_sync', False)
        self.slop = rospy.get_param('~slop', 0.5)
        size = tuple(rospy.get_param('~size', [640, 480]))
        assert len(size) == 2
        self.width, self.height = size