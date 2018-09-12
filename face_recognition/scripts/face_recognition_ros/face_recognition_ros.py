#! /usr/bin/env python
import cv2
import json
import numpy as np
import os
import openface
import rospy
import rospkg
import face_detector
import face_embosser
import face_classifier
from dlib import rectangle, rectangles

from cv_bridge import CvBridge
from sys import version_info
from sensor_msgs.msg import Image
from vision_system_msgs.msg import BoundingBox, FaceDescription, RecognizedFaces
from vision_system_msgs.srv import FaceClassifierTraining

BRIDGE = CvBridge()
PACK_DIR = rospkg.RosPack().get_path('face_recognition')

class FaceRecognitionROS():
    def __init__(self):
        self.models_dir = os.path.join(PACK_DIR, 'models')
        self.dataset_dir = os.path.join(PACK_DIR, 'dataset')
       
        self.num_recognitions = 0

        self.readParameters()
        self.face_detector = FaceDetector(self.detector_lib, self.aligner_lib)
        self.face_embosser = FaceEmbosser(self.embosser_lib)
        self.face_classifier = FaceClassifier(self.detector_lib)

    def readParameters(self):
        self.detector_lib = rospy.get_param('/face_recognition/detector/lib', 'opencv')

        self.opencv_detector_model = rospy.get_param('/face_recognition/detector/opencv/model', 'haarcascade_frontalface_default.xml')
        self.opencv_detector_cuda = rospy.get_param('/face_recognition/detector/opencv/cuda', True)


        self.aligner_lib = rospy.get_param('/face_recognition/aligner/lib', 'dlib')

        self.dlib_aligner_model = rospy.get_param('/face_recognition/aligner/dlib/model', 'shape_predictor_68_face_landmarks.dat')


        self.embosser_lib = rospy.get_param('/face_recognition/embosser/lib', 'facenet')

        self.facenet_embosser_model = rospy.get_param('/face_recognition/embosser/facenet/model', 'nn4.small2.v1.t7')
        self.facenet_embosser_cuda = rospy.get_param('/face_recognition/embosser/facenet/cuda', False)
        self.image_dimension = rospy.get_param('/face_recognition/embosser/facenet/image_dimension', 96)

        self.classifier_model = rospy.get_param('/face_recognition/classifier/model', 'classifier.pkl')
        self.classifier_threshold = rospy.get_param('/face_recognition/classifier/threshold', 0.5)

    def dlibRectangle2RosBoundingBox(self, rect):
        bounding_box = BoundingBox()
        bounding_box.minX = rect.tl_corner().x
        bounding_box.minY = rect.tl_corner().y
        bounding_box.width = rect.width()
        bounding_box.height = rect.height()
        return bounding_box

    def numpyndArray2dlibRectangles(self, array):
        rects = rectangles()
        for (x,y,w,h) in array:
            rects.append(rectangle(x, y, x + w, y + h))
        return rects

    def numpyArray2RosVector(self, array):
        vector = array.tolist()
        return vector

    def alignDataset(self):
        raw_dir = os.path.join(self.dataset_dir, 'raw')
        raw_aligned_dir = os.path.join(self.dataset_dir, 'raw_aligned')
        raw_labels = next(os.walk(raw_dir))[1]
        for lb in raw_labels:
            openface.helper.mkdirP(os.path.join(raw_aligned_dir, lb))

        raw_images = openface.data.iterImgs(raw_dir)
        raw_aligned_images = openface.data.iterImgs(raw_aligned_dir)

        ris = []
        rais = []

        for ri in raw_images:
            ris.append(ri)
        for rai in raw_aligned_images:
            rais.append(rai)

        images = []
        for ri in ris:
            found = False
            for rai in rais:
                if ri.name == rai.name and ri.cls == rai.cls:
                    found = True
            if not found:        
                images.append(ri)

        for image in images:
            rect = self.getLargestFaceBoundingBox(image.getRGB())
            aligned_face = self.alignFace(image.getRGB(), rect)
            bgr_image = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            print(os.path.join(raw_aligned_dir, image.cls, image.name + '.jpg'))
            cv2.imwrite(os.path.join(raw_aligned_dir, image.cls, image.name + '.jpg'), bgr_image)

    def generateDatasetFeatures(self):
        raw_aligned_dir = os.path.join(self.dataset_dir, 'raw_aligned')
        features_dir = os.path.join(self.dataset_dir, 'features')
        raw_aligned_images = openface.data.iterImgs(raw_aligned_dir)

        try:
            features_file = open(os.path.join(features_dir, 'features.json'), 'r+')
            features_data = json.load(features_file)
            features_file.truncate(0)
            features_file.close()
        except (OSError, IOError) as e:
            features_data = {}
    
        features_file = open(os.path.join(features_dir, 'features.json'), 'w')

        images = []
        for rai in raw_aligned_images:
            label = rai.cls
            name = rai.name
            if(label not in features_data.keys()):
                images.append(rai)
                features_data[label] = {}
            else:
                found = False
                for key in features_data[label].keys():
                    if(name == key):
                        found = True
                if(not found):
                    images.append(rai)

        for image in images:
            label = image.cls
            name = image.name
            features = self.extractFeaturesFromImage(image.getRGB())
            features_data[label][name] = self.numpyArray2RosVector(features)

        json.dump(features_data, features_file)

    def trainingProcess(self, ros_srv):
        self.alignDataset()
        self.generateDatasetFeatures()
        ans = self.trainClassifier(ros_srv.classifier_type, ros_srv.classifier_name)
        return ans

    def recognitionProcess(self, ros_msg):
        rgb_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="rgb8")
        face_rects = self.getAllFaceBoundingBoxes(rgb_image)
        faces_description = []
        if len(face_rects) == 0:
            return None
        for face_rect in face_rects:
            face_description = FaceDescription()
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
        recognized_faces.image_header = ros_msg.header

        recognized_faces.recognition_header.seq = self.num_recognitions
        recognized_faces.recognition_header.stamp = rospy.get_rostime()
        recognized_faces.recognition_header.frame_id = "face_recognition"

        recognized_faces.faces_description = faces_description

        self.num_recognitions += 1

        return recognized_faces