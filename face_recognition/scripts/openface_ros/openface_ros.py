#! /usr/bin/env python
import cv2
import json
import numpy as np
import os
import openface
import pickle
import rospy
import rospkg

from cv_bridge import CvBridge
from sys import version_info
from sensor_msgs.msg import Image
from vision_system_msgs.msg import BoundingBox, FaceDescription, RecognizedFaces
from vision_system_msgs.srv import FaceClassifierTraining

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

BRIDGE = CvBridge()
PACK_DIR = rospkg.RosPack().get_path('face_recognition')

class OpenfaceROS:
    def __init__(self):
        self.models_dir = os.path.join(PACK_DIR, 'models')

        self.dlib_model = ''
        self.openface_model = ''
        self.classifier_model = ''

        self.dataset_dir = os.path.join(PACK_DIR, 'dataset')
        self.image_dimension = 0
        self.threshold = 0.0
        self.cuda = False

        self.align = None
        self.net = None
        self.cl_label = None
        self.classifier = None

        self.num_recognitions = 0

        self.readParameters()
        self.createDlibAlign()
        self.createTorchNeuralNet()
        self.createClassifier()

    def readParameters(self):
        self.dlib_model = rospy.get_param('/face_recognition/dlib/model', 'shape_predictor_68_face_landmarks.dat')

        self.openface_model = rospy.get_param('/face_recognition/openface/model', 'nn4.small2.v1.t7')
        self.cuda = rospy.get_param('/face_recognition/openface/cuda', False)
        self.image_dimension = rospy.get_param('/face_recognition/dlib/image_dimension', 96)

        self.classifier_model = rospy.get_param('/face_recognition/classifier/model', 'classifier.pkl')
        self.threshold = rospy.get_param('/face_recognition/classifier/threshold', 0.5)

    def createDlibAlign(self):
        self.align = openface.AlignDlib(os.path.join(self.models_dir, 'dlib', self.dlib_model))

    def createTorchNeuralNet(self):
        self.net = openface.TorchNeuralNet(os.path.join(self.models_dir, 'openface', self.openface_model), self.image_dimension, cuda = self.cuda)

    def createClassifier(self):
        with open(os.path.join(self.models_dir, 'classifier', self.classifier_model), 'rb') as model_file:
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
        face_rect = self.align.getLargestFaceBoundingBox(image)
        return face_rect

    def alignFace(self, image, rect):
        aligned_face = self.align.align(self.image_dimension, image, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
        return aligned_face

    def extractFeaturesFromImage(self, image):
        feature_vector = self.net.forward(image)
        return feature_vector

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
        
    def trainClassifier(self, classifier_type, classifier_name):
        features_dir = os.path.join(self.dataset_dir, 'features')
        features_file = open(os.path.join(features_dir, 'features.json'), 'rw')
        features_data = json.load(features_file)

        labels = []
        embeddings = []
        for key in features_data.keys():
            labels += len(features_data[key].keys()) * [key]
            embeddings += features_data[key].values()

        embeddings = np.array(embeddings)

        le = LabelEncoder()
        le.fit(labels)
        labels_num = le.transform(labels)
        num_classes = len(le.classes_)

        if classifier_type == 'lsvm':
            clf = SVC(C=1, kernel='linear', probability=True)
        elif classifier_type == 'rsvm':
            clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif classifier_type == 'gssvm':
            param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
            ]
            clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif classifier_type == 'gmm':
            clf = GMM(n_components=num_classes)
        elif classifier_type == 'dt':
            clf = DecisionTreeClassifier(max_depth=20)
        elif classifier_type == 'gnb':
            clf = GaussianNB()
        else:
            return False

        clf.fit(embeddings, labels_num)
        fName = self.models_dir + '/classifier/' + classifier_name
        with open(fName, 'w') as f:
            pickle.dump((le, clf), f)
        return True

    def trainingProcess(self, ros_srv):
        self.alignDataset()
        self.generateDatasetFeatures()
        ans = self.trainClassifier(ros_srv.classifier_type, ros_srv.classifier_name)
        return ans

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
        recognized_faces.image_header = ros_msg.header

        recognized_faces.recognition_header.seq = self.num_recognitions
        recognized_faces.recognition_header.stamp = rospy.get_rostime()
        recognized_faces.recognition_header.frame_id = "face_recognition"

        recognized_faces.faces_description = faces_description

        self.num_recognitions += 1

        return recognized_faces