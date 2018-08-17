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
DIR = os.path.dirname(os.path.realpath(__file__))


class OpenfaceROS:
    def __init__(self):
        self.dlibmodel_dir = ''
        self.dlibmodel = ''
        self.openfacemodel_dir = ''
        self.openfacemodel = ''
        self.classifiermodel_dir = ''
        self.classifiermodel = ''
        self.dataset_dir = '' 
        self.image_dimension = 0
        self.threshold = 0.0

        self.align = None
        self.net = None
        self.cl_label = None
        self.classifier = None

        self.load()

    def load(self):
        config_file = open(os.path.join(DIR, 'config.json'), 'r')
        config_data = json.load(config_file)

        self.models_dir = os.path.join(DIR, 'models')
        self.dataset_dir = os.path.join(DIR, config_data['dataset_relative_dir'])

        self.image_dimension = config_data['image_dimension']
        self.threshold = config_data['threshold']

        self.createDlibAlign(config_data['dlib_model'])
        self.createTorchNeuralNet(config_data['openface_model'])
        self.createClassifier(config_data['classifier_model'])

    def createDlibAlign(self, dlib_model):
        self.align = openface.AlignDlib(os.path.join(self.models_dir, 'dlib', dlib_model))

    def createTorchNeuralNet(self, openface_model):
        self.net = openface.TorchNeuralNet(os.path.join(self.models_dir, 'openface', openface_model), self.image_dimension, cuda=True)

    def createClassifier(self, classifier_model):
        with open(os.path.join(self.models_dir, 'classifier', classifier_model), 'rb') as model_file:
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
            pass
        elif classifier_type == 'dt':
            clf = DecisionTreeClassifier(max_depth=20)
        elif classifier_type == 'gnb':
            clf = GaussianNB()
        elif classifier_type == 'dbn':
            from nolearn.dbn import DBN
            clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                learn_rates=0.3,
                # Smaller steps mean a possibly more accurate result, but the
                # training will take longer
                learn_rate_decays=0.9,
                # a factor the initial learning rate will be multiplied by
                # after each iteration of the training
                epochs=300,  # no of iternation
                # dropouts = 0.25, # Express the percentage of nodes that
                # will be randomly dropped as a decimal.
                verbose=1)
        else:
            pass

        clf.fit(embeddings, labels_num)
        fName = self.models_dir + '/classifier/' + classifier_name
        with open(fName, 'w') as f:
            pickle.dump((le, clf), f)

    def trainingProcess(self, ros_srv):
        self.alignDataset()
        self.generateDatasetFeatures()
        self.trainClassifier(ros_srv.classifier_type, ros_srv.classifier_name)
        return True

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
