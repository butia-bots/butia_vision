#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import numpy as np
import open3d as o3d

from butia_vision_bridge import VisionBridge

from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D, Description3D, Recognitions3D

import tf

class TwoD2ThreeD:
    def __init__(self):
        self.__readParameters()
        self.br = tf.TransformBroadcaster()
 
    # def __createDescriptionOpen3DPCD(self, array_point_cloud, data):
    #     if type==Description2D.DETECTION:
    #         pass
    #     elif type==Description2D.INSTANCE_SEGMENTATION or type==Description2D.SEMANTIC_SEGMENTATION:
    #         pass
    #     else:
    #         pass

    def __recognitions3DComputation(self, array_point_cloud, descriptions2d, header, pcd_header):
        output_data = Recognitions3D()
        output_data.header = header

        descriptions3d = []
        for d in descriptions2d:
            center_x, center_y = d.bbox.center.x, d.bbox.center.y
            size_x, size_y = d.bbox.size_x, d.bbox.size_y
            min_x = int(center_x - size_x/2)
            max_x = int(center_x + size_x/2)
            min_y = int(center_y - size_y/2)
            max_y = int(center_y + size_y/2)

            desc_point_cloud = array_point_cloud[min_y:max_y, min_x:max_x, :]
            pcd = VisionBridge.pointCloudArraystoOpen3D(desc_point_cloud[:, :, :3], desc_point_cloud[:, :, 3:])
            #center = pcd.get_center()
            #desc_point_cloud = desc_point_cloud.reshape(-1,6)
            #print(desc_point_cloud.shape)
            #print(np.mean(), axis=0).shape)
            #center = np.nanmean(desc_point_cloud, axis=0)
            #print(center)
            center = pcd.get_oriented_bounding_box().get_center()
            if center[0] != np.nan:
                self.br.sendTransform((center[0], center[1], center[2]),
                        tf.transformations.quaternion_from_euler(0, 0, 0),
                        pcd_header.stamp,
                        'test',
                        pcd_header.frame_id)


        output_data.descriptions = descriptions3d
        return output_data

    def recognitions2DtoRecognitions3D(self, data):
        header = data.header
        descriptions2d = data.descriptions
        pc2 = data.points
        img_depth = data.image_depth
        
        if pc2.width*pc2.height > 0:
            xyz, rgb = VisionBridge.pointCloud2XYZRGBtoArrays(pc2)
            array_point_cloud = np.append(xyz, rgb, axis=2)
            return self.__recognitions3DComputation(array_point_cloud, descriptions2d, header, pc2.header)
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