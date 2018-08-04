#! /usr/bin/env python
import cv2
import openface
import rospy
import json

from cv_bridge import CvBridge
from os import path

BRIDGE = CvBridge()

class OpenfaceROS:
    def __init__(self):
        self.openface_dir = ''
        self.dlibmodel_dir = ''
        self.netmodel_dir = ''

        self.image_dimension = 0

        self.align = None
        self.net = None

        self.load()

    def load(self):
        dir = path.dirname(path.realpath(__file__))

        config_file = open(path.join(dir, 'config.json'), 'r')
        config_data = json.load(config_file)

        self.openface_dir = config_data['dir']
        self.dlibmodel_dir = path.join(self.openface_dir, 'models', 'dlib', config_data['dlib_model'])
        self.netmodel_dir = path.join(self.openface_dir, 'models', 'openface', config_data['net_model'])
        self.image_dimension = config_data['image_dimension']

    def createDlibAlign(self):
        self.align = openface.AlignDlib(self.dlibmodel_dir)

    def createTorchNeuralNet(self):
        self.net = openface.TorchNeuralNet(self.netmodel_dir, self.image_dimension)

    def getAllFaceBoundingBoxes(self, ros_msg):
        cv_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="rgb8")
        face_bbs = self.align.getAllFaceBoundingBoxes(cv_image)
        return face_bbs
    
    def getLargestFaceBoundingBox(self, ros_msg):
        cv_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="CV_8UC3")
        face_bb = self.align.getAllFaceBoundingBoxes(cv_image)
        return face_bb
