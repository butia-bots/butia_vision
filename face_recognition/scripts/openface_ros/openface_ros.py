#! /usr/bin/env python
import cv2
import json
import numpy as np
import os
import openface
import pickle
import rospy

from cv_bridge import CvBridge
from sys import version_info
from vision_system_msgs.msg import BoundingBox, FaceDescription, RecognizedFaces
from sklearn.preprocessing import LabelEncoder

BRIDGE = CvBridge()

class OpenfaceROS:
    def __init__(self):
        self.dlibmodel_dir = ''
        self.dlibmodel = ''
        self.netmodel_dir = ''
        self.netmodel = ''
        self.classifymodel_dir = ''
        self.classifymodel = ''
        self.dataset_dir = '' 
        self.image_dimension = 0
        self.threshold = 0.0

        self.align = None
        self.net = None
        self.cl_label = None
        self.classifier = None

        self.load()

    def load(self):
        dir = os.path.dirname(os.path.realpath(__file__))

        config_file = open(os.path.join(dir, 'config.json'), 'r')
        config_data = json.load(config_file)

        self.dlibmodel_dir = os.path.join(dir, 'models', 'dlib')
        self.dlibmodel = config_data['dlib_model']
        self.netmodel_dir = os.path.join(dir, 'models', 'openface')
        self.netmodel = config_data['net_model']
        self.classifymodel_dir = os.path.join(dir, 'models', 'openface')
        self.classifymodel = config_data['classify_model']
        self.dataset_dir = os.path.join(dir, config_data['dataset_relative_dir'])
        self.image_dimension = config_data['image_dimension']
        self.threshold = config_data['threshold']

        self.createDlibAlign()
        self.createTorchNeuralNet()
        self.createClassifier()

    def createDlibAlign(self):
        self.align = openface.AlignDlib(os.path.join(self.dlibmodel_dir, self.dlibmodel))

    def createTorchNeuralNet(self):
        self.net = openface.TorchNeuralNet(os.path.join(self.netmodel_dir, self.netmodel), self.image_dimension, cuda=True)

    def createClassifier(self):
        with open(os.path.join(self.classifymodel_dir, self.classifymodel), 'rb') as model_file:
            if version_info[0] < 3:
                (self.cl_label, self.classifier) = pickle.load(model_file)
            else:
                (self.cl_label, self.classifier) = pickle.load(model_file, encoding='latin1')

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
        rep = array.reshape(1, -1)
        predictions = self.classifier.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = self.cl_label.inverse_transform(maxI)
        confidence = predictions[maxI]
        if confidence > self.threshold:
            return (person.decode('utf-8'), confidence)
        else:
            return ('Unknow', 0)

    def alignDataset(self):
        raw_dir = path.join(self.dataset_dir, 'raw'))
        raw_aligned_dir = path.join(self.dataset_dir, 'raw_aligned')
        
        raw_labels = next(os.walk(raw_dir))[1]
        for lb in raw_labels:
            openface.helper.mkdirP(path.join(raw_aligned_dir, lb))

        raw_images = openface.data.iterImgs(raw_dir)
        raw_aligned_images = openface.data.iterImgs(raw_aligned_dir)

        images = []
        for ri in raw_images:
            for rai in raw_aligned_images:
                if ri.name != rai.name or ri.cls != rai.cls:
                    images.append(ri)

        for image in images:
            rect = self.getLargestFaceBoundingBox(image.getRGB())
            aligned_face = self.alignFace(image.getRGB(), rect)
            cv2.imwrite(path.join(raw_aligned_dir, image.cls, image.name, '.jpg'), aligned_face)

    def trainClassifier(self, classifier_type):
        labels = next(os.walk(self.dataset_dir))[1]

        if classifier_type == 'lsvm':
            pass
        elif classifier_type == 'rsvm':
            pass
        elif classifier_type == 'gssvm':
            pass
        elif classifier_type == 'gmm':
            pass
        elif classifier_type == 'dt':
            pass
        elif classifier_type == 'gnb':
            pass
        elif classifier_type == 'dbn':
            pass
        else:
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
