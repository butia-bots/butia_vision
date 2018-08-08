#! /usr/bin/env python
import cv2
import openface
import rospy
import json
import numpy as np
import pickle

from cv_bridge import CvBridge
from sys import version_info
from os import path
from vision_system_msgs.msg import BoundingBox, FaceDescription, RecognizedFaces
from sklearn.preprocessing import LabelEncoder

BRIDGE = CvBridge()

class OpenfaceROS:
    def __init__(self):
        self.dlibmodel_dir = ''
        self.netmodel_dir = ''
        self.classifymodel_dir = ''
        self.image_dimension = 0
        self.threshold = 0.0

        self.align = None
        self.net = None

        self.load()

    def load(self):
        dir = path.dirname(path.realpath(__file__))

        config_file = open(path.join(dir, 'config.json'), 'r')
        config_data = json.load(config_file)

        self.dlibmodel_dir = path.join(dir, 'models', 'dlib', config_data['dlib_model'])
        self.netmodel_dir = path.join(dir, 'models', 'openface', config_data['net_model'])
        self.classifymodel_dir = path.join(dir, 'models', 'openface', config_data['classify_model'])
        self.image_dimension = config_data['image_dimension']
        self.threshold = config_data['threshold']

    def createDlibAlign(self):
        self.align = openface.AlignDlib(self.dlibmodel_dir)

    def createTorchNeuralNet(self):
        self.net = openface.TorchNeuralNet(self.netmodel_dir, self.image_dimension)

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

    def getAllFaceBoundingBoxes(self, image):
        faces_rect = self.align.getAllFaceBoundingBoxes(image)
        return faces_rect
    
    def getLargestFaceBoundingBox(self, image):
        face_rect = self.align.getAllFaceBoundingBoxes(image)
        return face_rect

    def alignFace(self, image, rect):
        aligned_face = self.align.align(self.image_dimension, image, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return aligned_face

    def extractFeaturesFromImage(self, image):
        feature_vector = self.net.forward(image)
        return feature_vector

    #funcao adaptada das demos do openface
    def classify(self, array):
        with open(self.classifymodel_dir, 'rb') as model_file:
            if version_info[0] < 3:
                (le, clf) = pickle.load(model_file)
            else:
                (le, clf) = pickle.load(model_file, encoding='latin1')

        rep = array.reshape(1, -1)
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if confidence > self.threshold:
            return (person.decode('utf-8'), confidence)
        else:
            return ('Unknow', 0)

    def trainClassifier(self):
        pass

    def recognitionProcess(self, ros_msg):
        rgb_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="rgb8")

        face_rects = self.getAllFaceBoundingBoxes(rgb_image)
        faces_description = []
        face_description = FaceDescription()
        if len(face_rects) == 0:
            return None
        for face_rect in face_rects:
            aligned_face = self.alignFace(rgb_image, face_rect)
            features_array = self.extractFeaturesFromImage(aligned_face)
            classification = self.classify(features_array)
            label_class = classification[0]
            confidence = classification[1]
            
            bounding_box = self.dlibRectangle2RosBoundingBox(face_rect)
            features = self.numpyArray2RosVector(features_array)

            face_description.label_class = label_class
            face_description.features = features
            face_description.probability = confidence
            face_description.bounding_box = bounding_box

            faces_description.append(face_description)

        recognized_faces = RecognizedFaces()
        recognized_faces.header = ros_msg.header
        recognized_faces.header.stamp = rospy.get_rostime()

        recognized_faces.faces_description = faces_description

        return recognized_faces
