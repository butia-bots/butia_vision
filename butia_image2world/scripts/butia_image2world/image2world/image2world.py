#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from operator import invert
from re import L
import rospy

import open3d as o3d
import numpy as np
import math

import ros_numpy

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

class Image2World:
    def __init__(self, init_node=False):
        self.__readParameters()

        self.DESCRIPTION_PROCESSING_ALGORITHMS = {
            Description2D.DETECTION: self.__detectionDescriptionProcessing,
            Description2D.INSTANCE_SEGMENTATION: self.__instanceSegmentationDescriptionProcessing,
            Description2D.SEMANTIC_SEGMENTATION: self.__semanticSegmentationDescriptionProcessing
        }

        self.debug = rospy.Publisher('pub/debug', PointCloud2, queue_size=1)
        self.marker_publisher = rospy.Publisher('pub/markers', MarkerArray, queue_size=self.queue_size)

        self.current_camera_info = None
        self.lut_table = None
    
    def __mountDescription3D(self, description2d, box, mean_color, header):
        description3d = Description3D()
        description3d.header = description2d.header
        description3d.poses_header = header
        description3d.id = description2d.id
        description3d.global_id = description2d.global_id
        description3d.label = description2d.label
        description3d.score = description2d.score

        box_center = box.get_center()
        box_size = box.get_max_bound() - box.get_min_bound()
        #box_r = box.R.copy()
        box_rotation = np.eye(4,4)
        #box_rotation[:3, :3] = box_r
        box_orientation = quaternion_from_matrix(box_rotation)
        box_size = np.dot(box_size, box_rotation[:3, :3])

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

        
        description3d.mean_color.r = mean_color[0]
        description3d.mean_color.g = mean_color[1]
        description3d.mean_color.b = mean_color[2]
        
        return description3d
    
    def __compareCameraInfo(self, camera_info):
        equal = True
        equal = equal and (camera_info.width == self.current_camera_info.width)
        equal = equal and (camera_info.height == self.current_camera_info.height)
        equal = equal and np.all(np.isclose(np.asarray(camera_info.K),
                                            np.asarray(self.current_camera_info.K)))
        return equal
        
    def __mountLutTable(self, camera_info):
        if self.lut_table is None or not self.__compareCameraInfo(camera_info):
            self.current_camera_info = camera_info
            K = np.asarray(camera_info.K).reshape((3,3))

            fx = 1./K[0,0]
            fy = 1./K[1,1]
            cx = K[0,2]
            cy = K[1,2]

            x_table = (np.arange(0, self.current_camera_info.width) - cx)*fx 
            y_table = (np.arange(0, self.current_camera_info.height) - cy)*fy

            x_mg, y_mg = np.meshgrid(x_table, y_table)

            self.lut_table = np.concatenate((x_mg[:, :, np.newaxis], y_mg[:, :, np.newaxis]), axis=2)
    
    def __detectionDescriptionProcessing(self, data, description2d, header):
        center_x, center_y = description2d.bbox.center.x, description2d.bbox.center.y
        bbox_size_x, bbox_size_y = description2d.bbox.size_x, description2d.bbox.size_y

        if bbox_size_x == 0 or bbox_size_y == 0:
            rospy.logwarn('BBox with zero size.')
            return None
        
        bbox_limits = [int(center_x - bbox_size_x/2), int(center_x + bbox_size_x/2), 
                       int(center_y - bbox_size_y/2), int(center_y + bbox_size_y/2)]
        
        if 'point_cloud' in data:
            array_point_cloud = data['point_cloud']

            w, h, _ = array_point_cloud.shape

            bbox_limits[0] = bbox_limits[0] if bbox_limits[0] > 0 else 0
            bbox_limits[1] = bbox_limits[1] if bbox_limits[1] > 0 else 0
            bbox_limits[2] = bbox_limits[2] if bbox_limits[2] < w else w-1
            bbox_limits[3] = bbox_limits[3] if bbox_limits[3] < h else h-1

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
                bbox_center = np.nanmedian(bbox_center_points, axis=0)
                if kernel_scale < 1.0 and kernel_scale + 0.1 > 1.0:
                    kernel_scale = 1.0
                else:
                    kernel_scale += 0.1

            if np.isnan(bbox_center).any():
                rospy.logwarn('BBox has no min valid points.')
                return None

            pcd = pcd.remove_non_finite_points()
            pcd = pcd.voxel_down_sample(self.voxel_grid_resolution)
            
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            [_, idx, _] = pcd_tree.search_knn_vector_3d(bbox_center, self.n_neighbors_cluster_selection)

            labels_array = np.asarray(pcd.cluster_dbscan(eps=self.voxel_grid_resolution*1.2, min_points=4))
            labels, counts = np.unique(labels_array, return_counts=True)

            labels_neigh_array = labels_array[idx]
            labels_neigh, counts_neigh = np.unique(labels_neigh_array, return_counts=True)
            counts_neigh[labels_neigh==-1] = 0

            desc_pcd = None
            query_count = 1
            while desc_pcd is None and query_count > 0:
                argmax = np.argmax(counts_neigh)
                query_count = counts_neigh[argmax]
                label = labels_neigh[argmax]
                if counts[labels==label] >= 4:
                    desc_pcd = pcd.select_by_index(list(np.argwhere(labels_array==label)))
                else:
                    counts_neigh[argmax] = 0

            box = desc_pcd.get_axis_aligned_bounding_box()

            mean_color = np.nanmean(np.asarray(desc_pcd.colors), axis=0)/255.0

            if desc_pcd is None:
                rospy.logwarn('Point Cloud has just noise. Try to increase voxel grid param.')
                return None
            
            description3d = self.__mountDescription3D(description2d, box, mean_color, header)

            description3d.raw_cloud = VisionBridge.arrays2toPointCloud2XYZRGB(np.asarray(pcd.points),
                                                                              np.asarray(pcd.colors), header)
            description3d.filtered_cloud = VisionBridge.arrays2toPointCloud2XYZRGB(np.asarray(desc_pcd.points),
                                                                                   np.asarray(desc_pcd.colors), header)

            if self.publish_debug:
                colors = np.asarray(desc_pcd.colors)
                colors[:, :] = np.array(self.color)
                
                self.debug.publish(VisionBridge.arrays2toPointCloud2XYZRGB(np.asarray(desc_pcd.points), colors, header))
        
        else:
            image_depth = data['image_depth']
            camera_info = data['camera_info']

            h, w = image_depth.shape

            bbox_limits[0] = bbox_limits[0] if bbox_limits[0] > 0 else 0
            bbox_limits[1] = bbox_limits[1] if bbox_limits[1] > 0 else 0
            bbox_limits[2] = bbox_limits[2] if bbox_limits[2] < w else w-1
            bbox_limits[3] = bbox_limits[3] if bbox_limits[3] < h else h-1

            self.__mountLutTable(camera_info)    

            center_depth = image_depth[int(center_y), int(center_x)]

            if center_depth == 0:
                rospy.logwarn('INVALID DEPTH VALUE')
            
            center_depth/= 1000.

            limits = np.asarray([(bbox_limits[0], bbox_limits[2]), (bbox_limits[1], bbox_limits[3])])

            vertices_3d = np.zeros((len(limits), 3))

            vertices_3d[:, :2] = self.lut_table[limits[:, 1], limits[:, 0], :]*center_depth
            vertices_3d[:, 2] = center_depth

            vertices_3d = np.concatenate((vertices_3d - np.array([0, 0, self.depth_mean_error]),
                                          vertices_3d + np.array([0, 0, 0.3])))

            min_bound = np.min(vertices_3d, axis=0)
            max_bound = np.max(vertices_3d, axis=0)

            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

            #TODO:
            '''
                - mount local point cloud using self.lut_table (from min_bound to max_bound)
                - put this local point cloud inside the description3D (may be useful to gerate gripper poses in manipulation)
                - generate a new bounding box using the points of the point cloud
            '''

            mean_color = np.array([0, 0, 0])

            description3d = self.__mountDescription3D(description2d, box, mean_color, header)

        return description3d


    def __semanticSegmentationDescriptionProcessing(self, source_data, description2d, header):
        return None

    def __instanceSegmentationDescriptionProcessing(self, source_data, description2d, header):
        return None

    def __createDescription3D(self, source_data, description2d, header):
        if description2d.type in self.DESCRIPTION_PROCESSING_ALGORITHMS:
            return self.DESCRIPTION_PROCESSING_ALGORITHMS[description2d.type](source_data, description2d, header)
        else:
            return None

    def __recognitions3DComputation(self, array_point_cloud, descriptions2d, recog_header, header):
        output_data = Recognitions3D()
        output_data.header = recog_header

        descriptions3d = [None]*len(descriptions2d)
        for i, d in enumerate(descriptions2d):
            descriptions3d[i] = self.__createDescription3D(array_point_cloud, d, header)

        output_data.descriptions = [d3 for d3 in descriptions3d if d3 is not None]
        return output_data

    def recognitions2DtoRecognitions3D(self, data):
        header = data.header
        descriptions2d = data.descriptions
        pc2 = data.points
        img_depth = data.image_depth
        camera_info = data.camera_info
        
        recognitions = Recognitions3D()
        if pc2.width*pc2.height > 0:
            xyz, rgb = VisionBridge.pointCloud2XYZRGBtoArrays(pc2)
            array_point_cloud = np.append(xyz, rgb, axis=2)
            data = {'point_cloud': array_point_cloud}
            recognitions = self.__recognitions3DComputation(data, descriptions2d, header, pc2.header)
        elif img_depth.width*img_depth.height > 0:
            if camera_info.width*camera_info.height == 0:
                rospy.logwarn('Feature not implemented: Image2World cannot use depth image without camera info as input.')
                return None
            imd = ros_numpy.numpify(img_depth)
            data = {'image_depth': imd, 'camera_info': camera_info}
            recognitions = self.__recognitions3DComputation(data, descriptions2d, header, img_depth.header)
            
        else:
            rospy.logwarn('Image2World cannot be used because pointcloud and depth images are void.')

        if self.publish_markers:
            self.publishMarkers(recognitions.descriptions)
        return recognitions

    def __callback(self, data):
        data_3d = self.recognitions2DtoRecognitions3D(data)
        if data_3d is not None:
            self.__publish(data_3d)

    def __publish(self, data):
        self.publisher.publish(data)

    def publishMarkers(self, descriptions3d):
        markers = MarkerArray()
        color = np.asarray(self.color)/255.0
        for i, det in enumerate(descriptions3d):
            name = det.label

            # cube marker
            marker = Marker()
            marker.header = det.poses_header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.4
            marker.ns = "bboxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.scale = det.bbox.size
            marker.lifetime = rospy.Time.from_sec(2)
            markers.markers.append(marker)

            # text marker
            marker = Marker()
            marker.header = det.poses_header
            marker.action = Marker.ADD
            marker.pose = det.bbox.center
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.id = i
            marker.ns = "texts"
            marker.type = Marker.TEXT_VIEW_FACING
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.text = '{} ({:.2f})'.format(name, det.score)
            marker.lifetime = rospy.Time.from_sec(2)
            markers.markers.append(marker)
        
        self.marker_publisher.publish(markers)
    
    def initROS(self):
        self.subscriber = rospy.Subscriber('sub/recognitions2d', Recognitions2D, self.__callback, queue_size=self.queue_size)
        self.publisher = rospy.Publisher('pub/recognitions3d', Recognitions3D, queue_size=self.queue_size)

    def __readParameters(self):
        self.queue_size = int(rospy.get_param('~queue_size', 1))
        self.kernel_scale = rospy.get_param('~kernel_scale', 0.1)
        self.kernel_min_size = int(rospy.get_param('~kernel_min_size', 5))
        self.voxel_grid_resolution = rospy.get_param('~voxel_grid_resolution', 0.03)
        self.n_neighbors_cluster_selection = int(rospy.get_param('~n_neighbors_cluster_selection', 5))
        self.publish_debug = rospy.get_param('~publish_debug', False)
        self.publish_markers = rospy.get_param('~publish_markers', True)
        self.color = rospy.get_param('~color', [255, 0, 0])
        self.depth_mean_error = rospy.get_param('~depth_mean_error', 0.017)

if __name__ == '__main__':
    rospy.init_node('image2world_node', anonymous = True)

    Image2World = Image2World()
    Image2World.initROS()

    rospy.spin()