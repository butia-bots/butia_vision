#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cmath import isnan
from itertools import count
import rospy

import numpy as np
import math

from butia_vision_bridge import VisionBridge

from sensor_msgs.msg import PointCloud2
from butia_vision_msgs.msg import Description2D, Recognitions2D, Description3D, Recognitions3D
from visualization_msgs.msg import Marker, MarkerArray

#from tf.transformations (that it is not working on jetson)
def quaternion_from_matrix(matrix):
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

class TwoD2ThreeD:
    def __init__(self, init_node=False):
        self.__readParameters()

        self.DESCRIPTION_PROCESSING_ALGORITHMS = {
            Description2D.DETECTION: self.__detectionDescriptionProcessing,
            Description2D.INSTANCE_SEGMENTATION: self.__instanceSegmentationDescriptionProcessing,
            Description2D.SEMANTIC_SEGMENTATION: self.__semanticSegmentationDescriptionProcessing
        }

        self.debug = rospy.Publisher('/debug', PointCloud2, queue_size=1)
        self.marker_publisher = rospy.Publisher('markers', MarkerArray, queue_size=self.queue_size)
    
    def __mountDescription3D(self, description2d, raw_cloud, filtered_cloud, pcd_header):
        description3d = Description3D()
        description3d.header = description2d.header
        description3d.poses_header = pcd_header
        description3d.id = description2d.id
        description3d.global_id = description2d.global_id
        description3d.label = description2d.label
        description3d.score = description2d.score

        box = filtered_cloud.get_oriented_bounding_box()
        box_center = box.get_center()
        box_size = box.get_max_bound() - box.get_min_bound()
        box_r = box.R.copy()
        box_rotation = np.eye(4,4)
        box_rotation[:3, :3] = box_r
        box_orientation = quaternion_from_matrix(box_rotation)
        box_size = np.dot(box_size, box_r)

        description3d.bbox.center.position.x = box_center[0]
        description3d.bbox.center.position.y = box_center[1]
        description3d.bbox.center.position.z = box_center[2]
        description3d.bbox.center.orientation.x = box_orientation[0]
        description3d.bbox.center.orientation.y = box_orientation[1]
        description3d.bbox.center.orientation.z = box_orientation[2]
        description3d.bbox.center.orientation.w = box_orientation[3]

        description3d.bbox.size.x = box_size[0]
        description3d.bbox.size.y = box_size[1]
        description3d.bbox.size.z = box_size[2]

        mean_color = np.nanmean(np.asarray(filtered_cloud.colors), axis=0)/255.0
        description3d.mean_color.r = mean_color[0]
        description3d.mean_color.g = mean_color[1]
        description3d.mean_color.b = mean_color[2]

        description3d.raw_cloud = VisionBridge.arrays2toPointCloud2XYZRGB(np.asarray(raw_cloud.points), np.asarray(raw_cloud.colors), pcd_header)
        description3d.filtered_cloud = VisionBridge.arrays2toPointCloud2XYZRGB(np.asarray(filtered_cloud.points), np.asarray(filtered_cloud.colors), pcd_header)

        colors = np.asarray(filtered_cloud.colors)
        colors[:, :] = np.array([255, 0, 0])
        
        self.debug.publish(VisionBridge.arrays2toPointCloud2XYZRGB(np.asarray(filtered_cloud.points), colors, pcd_header))
        
        # self.br.sendTransform((box_center[0], box_center[1], box_center[2]),
        #         box_orientation,
        #         pcd_header.stamp,
        #         'test',
        #         pcd_header.frame_id)
        
        return description3d

    def __detectionDescriptionProcessing(self, array_point_cloud, description2d, pcd_header):
        center_x, center_y = description2d.bbox.center.x, description2d.bbox.center.y
        bbox_size_x, bbox_size_y = description2d.bbox.size_x, description2d.bbox.size_y
        if bbox_size_x == 0 or bbox_size_y == 0:
            rospy.logwarn('BBox with zero size.')
            return None

        bbox_limits = (int(center_x - bbox_size_x/2), int(center_x + bbox_size_x/2), int(center_y - bbox_size_y/2), int(center_y + bbox_size_y/2))
        desc_point_cloud = array_point_cloud[bbox_limits[2]:bbox_limits[3], bbox_limits[0]:bbox_limits[1], :]
        pcd = VisionBridge.pointCloudArraystoOpen3D(desc_point_cloud[:, :, :3], desc_point_cloud[:, :, 3:])

        kernel_scale = self.kernel_scale
        bbox_center = np.array([np.nan, np.nan, np.nan])
        while np.isnan(bbox_center).any() and kernel_scale <= 1.0:
            size_x = int(bbox_size_x*kernel_scale)
            size_x = self.kernel_min_size if size_x < self.kernel_min_size else size_x
            size_y = int(bbox_size_y*kernel_scale)
            size_y = self.kernel_min_size if size_y < self.kernel_min_size else size_y
            center_limits = (int(center_x - size_x/2), int(center_x + size_x/2), int(center_y - size_y/2), int(center_y + size_y/2))
            bbox_center_points = array_point_cloud[center_limits[2]:center_limits[3], center_limits[0]:center_limits[1], :3].reshape(-1, 3)
            bbox_center = np.nanmean(bbox_center_points, axis=0)
            if kernel_scale < 1.0 and kernel_scale + 0.1 > 1.0:
                kernel_scale = 1.0
            else:
                kernel_scale += 0.1

        if np.isnan(bbox_center).any():
            rospy.logwarn('BBox has no min valid points.')
            return None

        pcd = pcd.remove_non_finite_points()
        pcd = pcd.voxel_down_sample(self.voxel_grid_resolution)
        labels_array = np.array(pcd.cluster_dbscan(eps=self.voxel_grid_resolution*1.2, min_points=4))
        labels, counts = np.unique(labels_array, return_counts=True)
        counts[labels==-1] = 0

        desc_pcd = None
        min_dist = float('inf')
        for label in labels:
            if label != -1:
                query_pcd = pcd.select_by_index(list(np.argwhere(labels_array==label)))
                query_center = query_pcd.get_center()
                query_dist = np.linalg.norm(query_center - bbox_center, ord=2)
                if query_dist < min_dist:
                    desc_pcd = query_pcd
                    min_dist = query_dist

        if desc_pcd is None or len(desc_pcd.points) < 4:
            rospy.logwarn('Point Cloud has just noise. Try to increase voxel grid param.')
            return None

        return self.__mountDescription3D(description2d, pcd, desc_pcd, pcd_header)


    def __semanticSegmentationDescriptionProcessing(self, array_point_cloud, description2d, pcd_header):
        return None

    def __instanceSegmentationDescriptionProcessing(self, array_point_cloud, description2d, pcd_header):
        return None

    def __createDescription3D(self, array_point_cloud, description2d, pcd_header):
        if description2d.type in self.DESCRIPTION_PROCESSING_ALGORITHMS:
            return self.DESCRIPTION_PROCESSING_ALGORITHMS[description2d.type](array_point_cloud, description2d, pcd_header)
        else:
            return None

    def __recognitions3DComputation(self, array_point_cloud, descriptions2d, header, pcd_header):
        output_data = Recognitions3D()
        output_data.header = header

        descriptions3d = [None]*len(descriptions2d)
        for i, d in enumerate(descriptions2d):
            descriptions3d[i] = self.__createDescription3D(array_point_cloud, d, pcd_header)

        output_data.descriptions = [d3 for d3 in descriptions3d if d3 is not None]
        return output_data

    def recognitions2DtoRecognitions3D(self, data):
        header = data.header
        descriptions2d = data.descriptions
        pc2 = data.points
        img_depth = data.image_depth
        
        recognitions = Recognitions3D()
        if pc2.width*pc2.height > 0:
            xyz, rgb = VisionBridge.pointCloud2XYZRGBtoArrays(pc2)
            array_point_cloud = np.append(xyz, rgb, axis=2)
            recognitions = self.__recognitions3DComputation(array_point_cloud, descriptions2d, header, pc2.header)
        elif img_depth.width*img_depth.height > 0:
            rospy.logwarn('Feature not implemented: TwoD2ThreeD cannot use depth image as input yet.')
        else:
            rospy.logwarn('TwoD2ThreeD cannot be used because pointcloud and depth images are void.')

        self.publishMarkers(recognitions.descriptions)
        return recognitions

    def __callback(self, data):
        data_3d = self.recognitions2DtoRecognitions3D(data)
        self.__publish(data_3d)

    def __publish(self, data):
        self.publisher.publish(data)

    def publishMarkers(self, descriptions3d):
        markers = MarkerArray()
        for i, det in enumerate(descriptions3d):
            name = det.label

            # cube marker
            marker = Marker()
            marker.header = det.poses_header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = 1.
            marker.color.g = 0
            marker.color.b = 0
            marker.color.a = 0.4
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = det.poses_header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = 1.
            marker.color.g = 0
            marker.color.b = 0
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.text = '{} ({:.2f})'.format(name, det.score)

            markers.markers.append(marker)
        
        self.marker_publisher.publish(markers)
    
    def initROS(self):
        self.subscriber = rospy.Subscriber('sub/recognitions2d', Recognitions2D, self.__callback, queue_size=self.queue_size)
        self.publisher = rospy.Publisher('pub/recognitions3d', Recognitions3D, queue_size=self.queue_size)

    def __readParameters(self):
        self.queue_size = int(rospy.get_param('~queue_size', 1))
        self.kernel_scale = rospy.get_param('~kernel_scale', 0.1)
        self.kernel_min_size = int(rospy.get_param('~kernel_min_size', 5))
        self.voxel_grid_resolution = rospy.get_param('~voxel_grid_resolution', 0.05)

if __name__ == '__main__':
    rospy.init_node('twoD2ThreeD_node', anonymous = True)

    twod2threed = TwoD2ThreeD()
    twod2threed.initROS()

    rospy.spin()