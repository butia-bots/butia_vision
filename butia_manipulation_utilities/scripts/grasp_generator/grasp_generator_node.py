#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from operator import invert
from re import L
import rospy
import os
import sys

import open3d as o3d
import numpy as np
import math
import time
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation

#import ros_numpy

#from butia_vision_bridge import VisionBridge

#from sensor_msgs.msg import PointCloud2
from butia_vision_msgs.msg import Recognitions2D, Recognitions3D, Description2D, GraspPose
from butia_vision_msgs.srv import GraspGenerator, GraspGeneratorResponse
from visualization_msgs.msg import Marker, MarkerArray

from contact_grasp_estimator import GraspEstimator
import config_utils

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

class GraspGeneratorNode:
    def __init__(self, init_node=False):
        self.__readParameters()
        self.initRosComm()

        self.marker_publisher = rospy.Publisher('pub/markers', MarkerArray, queue_size=self.queue_size)

        global_config = config_utils.load_config(self.ckpt_dir, batch_size=self.forward_passes, arg_configs=self.arg_configs)
        print(str(global_config))
        # Build the model
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, self.ckpt_dir, mode='test')

        self.bridge = CvBridge()
        self.main_vis = False
        self.tmp_img = ()
        self.tmp_grasp = ()

    def process_info(self, request):
        try:
            image_rgb = self.bridge.imgmsg_to_cv2(request.recognitions.image_rgb, "rgb8")
            cam_K = np.array(request.recognitions.camera_info.K).reshape(3, 3)
            image_depth = self.bridge.imgmsg_to_cv2(request.recognitions.image_depth, "passthrough").copy()
            image_depth[np.isnan(image_depth)] = 0.
            image_depth = image_depth/1000
            segmap = self.multiple_segmentations_to_image(request.recognitions)
            segmap_id = 1
            return self.inference(image_rgb, image_depth, segmap, cam_K, segmap_id)
        except CvBridgeError as e:
            rospy.logerr(e)

    def multiple_segmentations_to_image(self, data: Recognitions2D):
        """
        Converts multiple segmentations to a single mask
        """
        size = (data.descriptions[0].mask.height, data.descriptions[0].mask.width)
        final_image = np.zeros(size, dtype=np.uint8)
        for description in data.descriptions:
            if description.type == Description2D.SEMANTIC_SEGMENTATION:
                index = description.id
                mask = self.bridge.imgmsg_to_cv2(description.mask, desired_encoding="passthrough")
                for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i, j] > 0:  
                            final_image[i, j] = index+1
        return final_image  
    
    def transform_matrix_to_pose_vector(transform_matrix):
        """
        Converts a transformation matrix to a pose vector.

        Args:
            transform_matrix (numpy.ndarray): The transformation matrix to be converted.

        Returns:
            numpy.ndarray: The pose vector representing the translation and rotation of the transformation matrix.
        """
        translation = transform_matrix[:3, 3]
        rotation_matrix = transform_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()
        pose_vector = np.concatenate((translation, quaternion))
        return pose_vector

    def inference(self, rgb, depth, segmap, cam_K, segmap_id):
        begin = time.time()
        rospy.loginfo('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
            depth, cam_K, segmap=segmap, rgb=rgb,
            skip_border_objects=self.skip_border_objects, z_range=self.z_range)
        
        rospy.loginfo('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            self.sess, pc_full, pc_segments=pc_segments,
            local_regions=self.local_regions, filter_grasps=self.filter_grasps, forward_passes=self.forward_passes)
        rospy.loginfo('Grasps generated!')

        response = GraspGeneratorResponse()
        target_id = segmap_id if (self.local_regions or self.filter_grasps) else -1
        for k in pred_grasps_cam:
            if k != target_id:
                continue
            for pose_cam, score, contact_pt in zip(pred_grasps_cam[k], scores[k], contact_pts[k]):
                grasp_pose = GraspPose()
                grasp_pose.pred_grasps_cam = self.transform_matrix_to_pose_vector(pose_cam)
                grasp_pose.score = score
                grasp_pose.contact_pt = contact_pt
                response.grasp_poses.append(grasp_pose)

        # Visualize results
        pose_score = 0
        best_pose = GraspPose()
        for pose in response.grasp_poses:
            if pose_score < pose.score:
                pose_score = pose.score
                best_pose = pose
        print("Best pose: ", best_pose)

        print("inference time: ", time.time() - begin)

        return response

    
    def initRosComm(self):
        self.grasp_generator_srv = rospy.Service('butia_vision_msgs/grasp_generator', GraspGenerator, self.process_info)


    def __readParameters(self):
        self.queue_size = int(rospy.get_param('~queue_size', 1))
        self.publish_markers = rospy.get_param('~publish_markers', True)
        
        """
        Predict 6-DoF grasp distribution for given model and input data
        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param K: Camera Matrix with intrinsics to convert depth to point cloud
        :param local_regions: Crop 3D local regions around given segments. 
        :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
        :param filter_grasps: Filter and assign grasp contacts according to segmap.
        :param segmap_id: only return grasps from specified segmap_id.
        :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
        :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
        """
 
        self.ckpt_dir = rospy.get_param(
            "~ckpt_dir",
            os.path.dirname(os.path.abspath(__file__)).replace("/scripts/butia_manipulation_utilities", "") + "/checkpoints/scene_test_2048_bs3_hor_sigma_001")        
        self.z_range = rospy.get_param("~z_range", [0.2, 1.8])
        self.local_regions = rospy.get_param("~local_regions", True)
        self.filter_grasps = rospy.get_param("~filter_grasps", True)
        self.skip_border_objects = rospy.get_param("~skip_border_objects", False)
        self.visulize = rospy.get_param("~visulize", False)
        self.forward_passes = rospy.get_param("~forward_passes", 1)
        self.arg_configs = rospy.get_param("~arg_configs", [])
        
        global_config = config_utils.load_config(self.ckpt_dir, batch_size=self.forward_passes, arg_configs=self.arg_configs)
        print(str(global_config))

if __name__ == '__main__':
    rospy.init_node('grasp_generator_node', anonymous = True)
    
    grasp_generator = GraspGeneratorNode()

    rospy.spin()