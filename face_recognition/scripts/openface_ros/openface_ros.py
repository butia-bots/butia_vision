#! /usr/bin/env python
import cv2
import openface
import rospy
import json
import numpy

from cv_bridge import CvBridge
from os import path
from vision_system_msgs.msg import BoundingBox, FaceDescription, RecognizedFaces

BRIDGE = CvBridge()

class OpenfaceROS:
    def __init__(self):
        self.dlibmodel_dir = ''
        self.netmodel_dir = ''

        self.image_dimension = 0

        self.rgb_image = None

        self.align = None
        self.net = None

        self.load()

    def load(self):
        dir = path.dirname(path.realpath(__file__))

        config_file = open(path.join(dir, 'config.json'), 'r')
        config_data = json.load(config_file)

        self.dlibmodel_dir = path.join(dir, 'models', 'dlib', config_data['dlib_model'])
        self.netmodel_dir = path.join(dir, 'models', 'openface', config_data['net_model'])
        self.image_dimension = config_data['image_dimension']

    def createDlibAlign(self):
        self.align = openface.AlignDlib(self.dlibmodel_dir)

    def createTorchNeuralNet(self):
        self.net = openface.TorchNeuralNet(self.netmodel_dir, self.image_dimension)

    def carryImage(self, ros_msg):
        self.rgb_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="rgb8")

    def dlibRectangle2RosBoundingBox(self, rect):
        bounding_box = BoundingBox()
        bounding_box.minX = rect.tl_corner().x
        bounding_box.minY = rect.tl_corner().y
        bounding_box.width = rect.width()
        bounding_box.height = rect.height()
        return bounding_box

    def numpyArray2RosVector(self, array):
        vector = array.tolist()
        return vector

    def getAllFaceBoundingBoxes(self):
        faces_rect = self.align.getAllFaceBoundingBoxes(self.rgb_image)
        return faces_rect
    
    def getLargestFaceBoundingBox(self):
        face_rect = self.align.getAllFaceBoundingBoxes(self.rgb_image)
        return face_rect

    def alignFace(self, rect):
        aligned_face = self.align.align(self.image_dimension, self.rgb_image, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return aligned_face

    def extractFeaturesFromImage(self, image):
        feature_vector = self.net.forward(image)
        return feature_vector

    def clustering(self, array):
        return 'Unknow'

    def recognitionProcess(self, ros_msg):
        self.carryImage(ros_msg)

        self.createDlibAlign()
        self.createTorchNeuralNet()

        face_rects = self.getAllFaceBoundingBoxes()
        faces_description = []
        face_description = FaceDescription()
        if len(face_rects) == 0:
            return None
        for face_rect in face_rects:
            aligned_face = self.alignFace(face_rect)
            features_array = self.extractFeaturesFromImage(aligned_face)
            label_class = self.clustering(features_array)
            
            bounding_box = self.dlibRectangle2RosBoundingBox(face_rect)
            features = self.numpyArray2RosVector(features_array)

            face_description.label_class = label_class
            face_description.features = features
            face_description.bounding_box = bounding_box

            faces_description.append(face_description)

        recognized_faces = RecognizedFaces()
        recognized_faces.header = ros_msg.header
        recognized_faces.header.stamp = rospy.get_rostime()

        recognized_faces.faces_description = faces_description

        return recognized_faces
