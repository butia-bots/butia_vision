#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple
import rospy
import ros_numpy

import numpy as np
import cv2
#import open3d as o3d

from sensor_msgs.msg import Image, CameraInfo, PointCloud2

from multipledispatch import dispatch
from collections import Iterable

class VisionBridge:
    SOURCES_TYPES = {
        'camera_info': CameraInfo,
        'image_rgb': Image,
        'image_depth': Image,
        'points': PointCloud2
    }

    @dispatch(Image, Iterable)
    def reescaleROSData(data: Image, size: Iterable):
        width, height = size
        header = data.header
        encoding = data.encoding
        image = ros_numpy.numpify(data)
        image = cv2.resize(image, (width, height), cv2.INTER_LINEAR)

        data = ros_numpy.msgify(Image, image, encoding)
        data.header = header
        return data
    
    # just focal distances and optic centers must be rescaled
    @dispatch(CameraInfo, Iterable)
    def reescaleROSData(data: CameraInfo, size: Iterable):
        width, height = size
        scale_x = width/data.width
        scale_y = height/data.height

        data.K[0] *= scale_x
        data.K[2] *= scale_x
        data.K[4] *= scale_y
        data.K[5] *= scale_y

        data.width = width
        data.height = height

        return data

    '''
        # this function took some time to made. The hardest part is to transform a number called 'rgb' that is a float32 in three uint8
        # note the difference in the use of numpy 'view' and numpy 'astype':
            - using .astype, the number is simple converted to the other type, as like: 32.0 (float) -> 32 (int)
            - using .view, the binary of the number is viewed in another type, so the binary of a float32 can be read as a binary of an uint32.
        # It is needed to evaluate if this is a problem, but the resize (downsample or upsample) of the point cloud is done in a separated way. First in points, and after in colors.
    '''
    def pointCloudArraystoOpen3D(xyz, rgb):
        if len(xyz.shape) == 3:
            xyz = xyz.reshape(-1, 3)
        if len(rgb.shape) == 3:
            rgb = rgb.reshape(-1, 3)
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(xyz)
        #pcd.colors = o3d.utility.Vector3dVector(rgb)
        #pcd.remove_non_finite_points()
        return None

    def pointCloud2XYZRGBtoOpen3D(data: PointCloud2):
        xyz, rgb = VisionBridge.pointCloud2XYZRGBtoArrays(data)
        pcd = VisionBridge.pointCloudArraystoOpen3D(xyz, rgb)

        return pcd

    def pointCloud2XYZRGBtoArrays(data: PointCloud2):
        pc = ros_numpy.numpify(data)
        xyz = np.zeros((pc.shape[0], pc.shape[1], 3), dtype=np.float)
        rgb = np.zeros((pc.shape[0], pc.shape[1], 3), dtype=np.uint32)
        xyz[:, :, 0] = pc['x']
        xyz[:, :, 1] = pc['y']
        xyz[:, :, 2] = pc['z']
        rgb[:, :, 0] = (pc['rgb'].view(np.uint32) >> 16 & 255).astype(np.uint32)
        rgb[:, :, 1] = (pc['rgb'].view(np.uint32) >> 8 & 255).astype(np.uint32)
        rgb[:, :, 2] = (pc['rgb'].view(np.uint32) & 255).astype(np.uint32)

        return xyz, rgb
    
    def arrays2toPointCloud2XYZRGB(xyz, rgb, header):
        if len(xyz.shape) == 3:
            pc = np.zeros((xyz.shape[0], xyz.shape[1]), dtype={'names':('x', 'y', 'z', 'rgb'), 'formats':('f4', 'f4', 'f4', 'f4')})
            pc['x'] = xyz[:, :, 0]
            pc['y'] = xyz[:, :, 1]
            pc['z'] = xyz[:, :, 2]
            pc['rgb'] = (rgb[:, :, 0].astype(np.uint32) << 16 | rgb[:, :, 1].astype(np.uint32) << 8 | rgb[:, :, 2].astype(np.uint32)).view(np.float32)
        elif len(xyz.shape) == 2:
            pc = np.zeros((xyz.shape[0]), dtype={'names':('x', 'y', 'z', 'rgb'), 'formats':('f4', 'f4', 'f4', 'f4')})
            pc['x'] = xyz[:, 0]
            pc['y'] = xyz[:, 1]
            pc['z'] = xyz[:, 2]
            pc['rgb'] = (rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2].astype(np.uint32)).view(np.float32)
        else:
            return None
        data = ros_numpy.msgify(PointCloud2, pc, stamp=header.stamp, frame_id=header.frame_id)
        
        return data

    @dispatch(PointCloud2, Iterable)
    def reescaleROSData(data: PointCloud2, size: Iterable):
        width, height = size
        header = data.header
        xyz, rgb = VisionBridge.pointCloud2XYZRGBtoArrays(data)
        points = np.append(xyz, rgb.astype(np.float), axis=2)
        points = cv2.resize(points, (width, height), cv2.INTER_LINEAR)
        data = VisionBridge.arrays2toPointCloud2XYZRGB(points[:, :, :3], points[:, :, 3:], header)
        return data

    def initROS(self):
        rospy.init_node('butia_vision_bridge_node', anonymous = True)

        self.__readParameters()
        
        self.publishers = {}
        for source in VisionBridge.SOURCES_TYPES:
            rospy.Subscriber('sub/' + source, VisionBridge.SOURCES_TYPES[source], callback=self.__callback, callback_args=(source), queue_size=self.queue_size)
            self.publishers[source] = rospy.Publisher('pub/' + source, VisionBridge.SOURCES_TYPES[source], queue_size=self.queue_size)

        rospy.spin()

    def __init__(self, init_ros=True):
        if init_ros:
            self.initROS()

    def __callback(self, data, source):
        data = VisionBridge.reescaleROSData(data, (self.width, self.height))
        self.__publish(data, source)

    def __publish(self, data, source):
        self.publishers[source].publish(data)
    
    def __readParameters(self):
        self.queue_size = int(rospy.get_param('~queue_size', 1))
        size = tuple(rospy.get_param('~size', [640, 480]))
        assert len(size) == 2
        self.width = int(size[0])
        self.height = int(size[1])


if __name__ == '__main__':
    vision_bridge = VisionBridge()