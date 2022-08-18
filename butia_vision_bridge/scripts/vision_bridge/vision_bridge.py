#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import ros_numpy

import numpy as np
import cv2

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from multipledispatch import dispatch

class VisionBridge:
    SOURCES_TYPES = {
        'camera_info': CameraInfo,
        'image_rgb': Image,
        'image_depth': Image,
        'points': PointCloud2
    }

    def __init__(self):
        self.readParameters()

        self.createSubscribersAndPublishers()
    
    @dispatch(Image)
    def processData(self, data: Image):
        encoding = data.encoding
        image = ros_numpy.numpify(data)
        image = cv2.resize(image, (self.width, self.height), cv2.INTER_LINEAR)

        data = ros_numpy.msgify(Image, image, encoding)
        return data
    
    # just focal distances and optic centers must be rescaled
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

    '''
        # this function took some time to made. The hardest part is to transform a number called 'rgb' that is a float32 in three uint8
        # note the difference in the use of numpy 'view' and numpy 'astype':
            - using .astype, the number is simple converted to the other type, as like: 32.0 (float) -> 32 (int)
            - using .view, the binary of the number is viewed in another type, so the binary of a float32 can be read as a binary of an uint32.
        # It is needed to evaluate if this is a problem, but the resize (downsample or upsample) of the point cloud is done in a separated way. First in points, and after in colors.
    '''
    @dispatch(PointCloud2)
    def processData(self, data: PointCloud2):
        header = data.header
        pc = ros_numpy.numpify(data)
        points = np.zeros((pc.shape[0], pc.shape[1], 6), dtype=np.float)
        points[:, :, 0] = pc['x']
        points[:, :, 1] = pc['y']
        points[:, :, 2] = pc['z']
        points[:, :, 3] = (pc['rgb'].view(np.uint32) >> 16 & 255).astype(np.float)
        points[:, :, 4] = (pc['rgb'].view(np.uint32) >> 8 & 255).astype(np.float)
        points[:, :, 5] = (pc['rgb'].view(np.uint32) & 255).astype(np.float)
        points = cv2.resize(points, (self.width, self.height), cv2.INTER_LINEAR)
        pc = np.zeros((points.shape[0], points.shape[1]), dtype={'names':('x', 'y', 'z', 'rgb'), 'formats':('f4', 'f4', 'f4', 'f4')})
        pc['x'] = points[:, :, 0]
        pc['y'] = points[:, :, 1]
        pc['z'] = points[:, :, 2]
        pc['rgb'] = (points[:, :, 3].astype(np.uint32) << 16 | points[:, :, 4].astype(np.uint32) << 8 | points[:, :, 5].astype(np.uint32)).view(np.float32)

        data = ros_numpy.msgify(PointCloud2, pc, stamp=header.stamp, frame_id=header.frame_id)
        return data

    def callback(self, data, source):
        data = self.processData(data)
        self.publish(data, source)

    def publish(self, data, source):
        self.publishers[source].publish(data)
    
    def createSubscribersAndPublishers(self):
        self.publishers = {}
        for source in VisionBridge.SOURCES_TYPES:
            rospy.Subscriber('sub/' + source, VisionBridge.SOURCES_TYPES[source], callback=self.callback, callback_args=(source), queue_size=self.queue_size)
            self.publishers[source] = rospy.Publisher('pub/' + source, VisionBridge.SOURCES_TYPES[source], queue_size=self.queue_size)

    def readParameters(self):
        self.queue_size = int(rospy.get_param('~queue_size', 1))
        size = tuple(rospy.get_param('~size', [640, 480]))
        assert len(size) == 2
        self.width = int(size[0])
        self.height = int(size[1])