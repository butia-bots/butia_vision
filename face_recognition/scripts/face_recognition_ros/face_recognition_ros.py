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

        self.detector_lib = ''
        self.aligner_lib = ''
        self.embosser_lib = ''
        self.classifier_lib = ''

        self.loads_dict = {}
        self.detectors_dict = {}
        self.aligners_dict = {}
        self.embossers_dict = {}
        self.classifiers_dict = {}

        self.readParameters()
        self.face_detector = self.loads_dict[self.detector_lib]
        self.face_aligner = self.loads_dict[self.aligner_lib]
        self.face_embosser = self.loads_dict[self.embosser_lib]
        self.face_classifier = self.loads_dict[self.classifer_lib]

    def readParameters(self):
        self.detector_lib = rospy.get_param('/face_recognition/detector/lib', 'opencv')

        opencv_detector_model = rospy.get_param('/face_recognition/detector/opencv/model', 'haarcascade_frontalface_default.xml')
        opencv_detector_cuda = rospy.get_param('/face_recognition/detector/opencv/cuda', True)
        opencv_detector_scale_factor = rospy.get_param('/face_recognition/detector/opencv/scale_factor', 1.3)
        opencv_detector_min_neighbors = rospy.get_param('/face_recognition/detector/opencv/min_neighbors', 5)

        self.loads_dict['opencv'] = {
            'function' : loadOpencvModels,
            'args' : (self.models_dir),
            'kwargs' : {
                'model' : opencv_detector_model,
                'cuda' : opencv_detector_cuda    
            }
        }
        self.detectors_dict['opencv'] = {
            'function' : detectFacesOpencv,
            'args' : (),
            'kwargs' : {}
        }
        self.detectors_dict['dlib'] = {
            'function' : detectFacesDlib,
            'args' : (),
            'kwargs' : {}
        }
        

        self.aligner_lib = rospy.get_param('/face_recognition/aligner/lib', 'dlib')

        dlib_aligner_model = rospy.get_param('/face_recognition/aligner/dlib/model', 'shape_predictor_68_face_landmarks.dat')
        dlib_aligner_image_dimension = rospy.get_param('/face_recognition/aligner/dlib/image_dimension', 96)

        self.loads_dict['dlib'] = {
            'function' : loadDlibModels,
            'args' : (self.models_dir),
            'kwargs' : {
                'model' : dlib_aligner_model   
            }
        }
        self.aligners_dict['dlib'] = {
            'function' : alignFace,
            'args' : (),
            'kwargs' : {
                'image_dimension' : dlib_aligner_image_dimension
            }
        }


        self.embosser_lib = rospy.get_param('/face_recognition/embosser/lib', 'facenet')

        facenet_embosser_model = rospy.get_param('/face_recognition/embosser/facenet/model', 'nn4.small2.v1.t7')
        facenet_embosser_image_dimension = dlib_aligner_image_dimension
        facenet_embosser_cuda = rospy.get_param('/face_recognition/embosser/facenet/cuda', False)

        self.loads_dict['facenet'] = {
            'function' : loadFacenetModels,
            'args' : (self.models_dir),
            'kwargs' : {
                'model' : facenet_embosser_model,
                'image_dimension' : facenet_embosser_image_dimension,
                'cuda' : facenet_embosser_cuda 
            }
        }
        self.embossers_dict['facenet'] = {
            'function' : extractFeaturesFacenet,
            'args' : (),
            'kwargs' : {}
        }

        self.classifier_lib = rospy.get_param('/face_recognition/classifier/lib', 'sklearn')

        sklearn_classifier_model = rospy.get_param('/face_recognition/classifier/sklearn/model', 'classifier.pkl')
        sklearn_classifier_threshold = rospy.get_param('/face_recognition/classifier/sklearn/threshold', 0.5)

        self.loads_dict['sklearn'] = {
            'function' : loadSklearnModels,
            'args' : (self.models_dir),
            'kwargs' : {
                'model' : sklearn_classifier_model 
            }
        }
        self.classifiers_dict['sklearn'] = {
            'function' : classifySklearn,
            'args' : (),
            'kwargs' : {
                'threshold' : sklearn_classifier_threshold
            }
        }


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
        ans = trainClassifier(ros_srv.classifier_type, ros_srv.classifier_name)
        return ans

    def recognitionProcess(self, ros_msg):
        rgb_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="rgb8")

        detector_dict = self.detectors_dict[self.detector_lib]
        face_rects = detector_dict['function'](self.face_detector, rgb_image, *detector_dict['args'], **detector_dict['kwargs'])
        faces_description = []
        if len(face_rects) == 0:
            return None
        for face_rect in face_rects:
            face_description = FaceDescription()

            aligner_dict = self.aligners_dict[self.aligner_lib]
            aligned_face = aligner_dict['function'](rgb_image, face_rect, *aligner_dict['args'], **aligner_dict['kwargs'])

            embosser_dict = self.embossers_dict[self.embosser_lib]
            features_array = embosser_dict['function'](aligned_face, *embosser_dict['args'], **embosser_dict['kwargs'])

            classifier_dict = self.classifiers_dict[self.classifier_lib]
            classification = classifier_dict['function'](features_array, *classifier_dict['args'], **classifier_dict['kwargs'])
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