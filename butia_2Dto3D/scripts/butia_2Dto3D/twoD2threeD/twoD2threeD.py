#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from email import header
import rospy

import numpy as np
import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh

from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D, Description3D, Recognitions3D

class TwoD2ThreeD:
    def __init__(self):
        self.__readParameters()
    
    def __createDescriptionPCD(self, pcd, type, bbox=BoundingBox2D(), mask=Image()):
        if type==Description2D.DETECTION:
            pass
        elif type==Description2D.INSTANCE_SEGMENTATION or type==Description2D.SEMANTIC_SEGMENTATION:
            pass
        else:
            pass

    def __open3dRecognitions3DComputation(self, pcd, descriptions2d, header, pcd_header):
        output_data = Recognitions3D()
        output_data.header = header

        descriptions3d = []
        for d in descriptions2d:
            pass
        output_data.descriptions = descriptions3d
        return output_data

    def recognitions2DtoRecognitions3D(self, data):
        header = data.header
        descriptions2d = data.descriptions
        pc2 = data.points
        img_depth = data.image_depth
        
        if pc2.width*pc2.height > 0:
            pcd = orh.rospc_to_o3dpc(pc2)
            return self.__open3dRecognitions3DComputation(pcd, descriptions2d, header, pc2.header)
        elif img_depth.width*img_depth.height > 0:
            rospy.logwarn('Feature not implemented: TwoD2ThreeD cannot use depth image as input yet.')
            return Recognitions3D()
        else:
            rospy.logwarn('TwoD2ThreeD cannot be used because pointcloud and depth images are void.')
            return Recognitions3D()   

    def __callback(self, data):
        data_3d = self.recognitions2DtoRecognitions3D(data)
        self.__publish(data_3d)

    def __publish(self, data):
        self.publisher.publish(data)
    
    def initROS(self):
        self.subscriber = rospy.Subscriber('sub/recognitions2d', Recognitions2D, self.__callback, queue_size=self.queue_size)
        self.publisher = rospy.Publisher('pub/recognitions3d', Recognitions3D, queue_size=self.queue_size)

    def __readParameters(self):
        self.queue_size = int(rospy.get_param('~queue_size', 1))

if __name__ == '__main__':
    rospy.init_node('twoD2ThreeD_node', anonymous = True)

    twod2threed = TwoD2ThreeD()
    twod2threed.initROS()

    rospy.spin()