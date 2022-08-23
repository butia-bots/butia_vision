#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cmath import isnan
import rospy

import numpy as np
import open3d as o3d

from butia_vision_bridge import VisionBridge

from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from sensor_msgs.msg import Image, PointCloud2
from butia_vision_msgs.msg import Description2D, Recognitions2D, Description3D, Recognitions3D

import tf

class TwoD2ThreeD:
    def __init__(self):
        self.__readParameters()
        self.br = tf.TransformBroadcaster()
        self.debug = rospy.Publisher('/debug', PointCloud2)
 
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
            bbox_size_x, bbox_size_y = d.bbox.size_x, d.bbox.size_y
            if bbox_size_x == 0 or bbox_size_y == 0:
                rospy.logwarn('BBox with zero size.')
                continue

            bbox_limits = (int(center_x - bbox_size_x/2), int(center_x + bbox_size_x/2), int(center_y - bbox_size_y/2), int(center_y + bbox_size_y/2))
            desc_point_cloud = array_point_cloud[bbox_limits[2]:bbox_limits[3], bbox_limits[0]:bbox_limits[1], :]
            pcd = VisionBridge.pointCloudArraystoOpen3D(desc_point_cloud[:, :, :3], desc_point_cloud[:, :, 3:])

            kernel_scale = self.kernel_scale
            bbox_center = np.array([np.nan, np.nan, np.nan])
            while np.isnan(bbox_center).any() and kernel_scale <= 1:
                size_x = int(bbox_size_x*kernel_scale)
                size_x = self.kernel_min_size if size_x < self.kernel_min_size else size_x
                size_y = int(bbox_size_y*kernel_scale)
                size_y = self.kernel_min_size if size_y < self.kernel_min_size else size_y
                center_limits = (int(center_x - size_x/2), int(center_x + size_x/2), int(center_y - size_y/2), int(center_y + size_y/2))
                bbox_center_points = array_point_cloud[center_limits[2]:center_limits[3], center_limits[0]:center_limits[1], :3]
                bbox_center = np.nanmean(bbox_center_points.reshape(-1, 3), axis=0)
                kernel_scale += 0.1

            bbox_center_distance = np.nanmean(np.linalg.norm((bbox_center_points.reshape(-1, 3) - bbox_center), axis=1, ord=2))

            if np.isnan(bbox_center).any():
                rospy.logwarn('BBox has only nan or inf values.')
                continue

            pcd.remove_non_finite_points()
            labels_array = np.array(pcd.cluster_dbscan(eps=bbox_center_distance, min_points=20))
            labels, counts = np.unique(labels_array, return_counts=True)
            max_label = labels[np.argmax(counts)]
    
            desc_indices = np.argwhere(labels_array==max_label)

            pcd = pcd.select_by_index(list(desc_indices))

            colors = np.asarray(pcd.colors)
            colors[:, :] = np.array([255, 0, 0])
            
            self.debug.publish(VisionBridge.arrays2toPointCloud2XYZRGB(np.asarray(pcd.points), colors, pcd_header))

            aabb = pcd.get_axis_aligned_bounding_box()
            
            center = aabb.get_center()
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
        self.kernel_scale = rospy.get_param('~kernel_scale', 0.1)
        self.kernel_min_size = int(rospy.get_param('~kernel_min_size', 1))

if __name__ == '__main__':
    rospy.init_node('twoD2ThreeD_node', anonymous = True)

    twod2threed = TwoD2ThreeD()
    twod2threed.initROS()

    rospy.spin()