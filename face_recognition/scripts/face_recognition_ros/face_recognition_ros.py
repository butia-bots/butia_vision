#! /usr/bin/env python
import cv2
import json
import numpy as np
import os
import openface
import rospy
import rospkg
from face_detector import *
from face_aligner import *
from face_embosser import *
from face_classifier import *
from dlib import rectangle, rectangles

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_system_msgs.msg import BoundingBox, Description, Description3D, Recognitions, Recognitions3D
from vision_system_msgs.srv import FaceClassifierTraining

BRIDGE = CvBridge()
PACK_DIR = rospkg.RosPack().get_path('face_recognition')

class FaceRecognitionROS():
    def __init__(self):
        self.models_dir = os.path.join(PACK_DIR, 'models')
        self.dataset_dir = os.path.join(PACK_DIR, 'dataset')
       
        self.last_recognition = 0.0

        self.image_width = 0
        self.image_height = 0

        self.detector_model_id = ''
        self.aligner_model_id = ''
        self.embosser_model_id = ''
        self.classifier_model_id = ''

        self.detectors_dict = {}
        self.aligners_dict = {}
        self.embossers_dict = {}
        self.classifiers_dict = {}

        self.readParameters()

        self.loadDetector()
        self.loadAligner()
        self.loadEmbosser()
        self.loadClassifier()
        
    def readParameters(self):
        self.verbose = rospy.get_param('/face_recognition/verbose', True)
        self.debug = rospy.get_param('/face_recognition/debug', False)

        self.detector_model_id = rospy.get_param('/face_recognition/detector/model_id', 'opencv_dnn')
        self.mountDetectorsDict()

        self.aligner_model_id = rospy.get_param('/face_recognition/aligner/model_id', 'dlib')
        self.mountAlignersDict()
        

        self.embosser_model_id = rospy.get_param('/face_recognition/embosser/model_id', 'facenet')
        self.mountEmbossersDict()
       

        self.classifier_model_id = rospy.get_param('/face_recognition/classifier/model_id', 'sklearn')
        self.mountClassifiersDict()
       

    def mountDetectorsDict(self):
        opencv_cascade_detector_model = rospy.get_param('/face_recognition/detector/opencv_cascade/model', 'haarcascade_frontalface_default.xml')
        opencv_cascade_detector_cuda = rospy.get_param('/face_recognition/detector/opencv_cascade/cuda', True)
        opencv_cascade_detector_scale_factor = rospy.get_param('/face_recognition/detector/opencv_cascade/scale_factor', 1.3)
        opencv_cascade_detector_min_neighbors = rospy.get_param('/face_recognition/detector/opencv_cascade/min_neighbors', 5)
        opencv_cascade_detector_height = rospy.get_param('/face_recognition/detector/opencv_cascade/height', 300)

        self.detectors_dict['opencv_cascade'] = {
            'load' : {
                'function' : loadOpencvCascadeModel,
                'args' : (self.models_dir,),
                'kwargs' : {
                    'model' : opencv_cascade_detector_model,
                    'cuda' : opencv_cascade_detector_cuda,
                    'debug' : self.debug    
                }
            },
            'action' : {
                'function' : detectFacesOpencvCascade,
                'args' : (),
                'kwargs' : {
                    'scale_factor' : opencv_cascade_detector_scale_factor,
                    'min_neighbors' : opencv_cascade_detector_min_neighbors,
                    'height' : opencv_cascade_detector_height, 
                    'debug' : self.debug,
                    'verbose' : self.verbose
                }
            }
        }

        opencv_dnn_detector_model = rospy.get_param('/face_recognition/detector/opencv_dnn/model', 'tensorflow')
        opencv_dnn_detector_threshold = rospy.get_param('/face_recognition/detector/opencv_dnn/threshold', 0.7)
        opencv_dnn_detector_scale_factor = rospy.get_param('/face_recognition/detector/opencv_dnn/scale_factor', 1.0)
        opencv_dnn_detector_height = rospy.get_param('/face_recognition/detector/opencv_dnn/height', 300)
        opencv_dnn_detector_mean = rospy.get_param('/face_recognition/detector/opencv_dnn/mean', [104, 117, 123])

        self.detectors_dict['opencv_dnn'] = {
            'load' : {
                'function' : loadOpencvDnnModel,
                'args' : (self.models_dir,),
                'kwargs' : {
                    'model' : opencv_dnn_detector_model,
                    'debug' : self.debug    
                }
            },
            'action' : {
                'function' : detectFacesOpencvDnn,
                'args' : (),
                'kwargs' : {
                    'threshold' : opencv_dnn_detector_threshold,
                    'scale_factor' : opencv_dnn_detector_scale_factor,
                    'height' : opencv_dnn_detector_height,
                    'mean' : opencv_dnn_detector_mean,
                    'debug' : self.debug,
                    'verbose' : self.verbose
                }
            }
        }


        dlib_hog_detector_model = rospy.get_param('/face_recognition/detector/dlib_hog/model', 'default')
        dlib_hog_detector_height = rospy.get_param('/face_recognition/detector/dlib_hog/height', 300)

        self.detectors_dict['dlib_hog'] = {
            'load' : {
                'function' : loadDlibHogModel,
                'args' : (self.models_dir,),
                'kwargs' : {
                    'model' : dlib_hog_detector_model,
                    'debug' : self.debug    
                }
            },
            'action' : {
                'function' : detectFacesDlibHog,
                'args' : (),
                'kwargs' : {
                    'height' : dlib_hog_detector_height,
                    'debug' : self.debug,
                    'verbose' : self.verbose
                }
            }
        }


        dlib_mmod_detector_model = rospy.get_param('/face_recognition/detector/dlib_mmod/model', 'mmod_human_face_detector.dat')
        dlib_mmod_detector_height = rospy.get_param('/face_recognition/detector/dlib_mmod/height', 300)

        self.detectors_dict['dlib_mmod'] = {
            'load' : {
                'function' : loadDlibMmodModel,
                'args' : (self.models_dir,),
                'kwargs' : {
                    'model' : dlib_mmod_detector_model,
                    'debug' : self.debug    
                }
            },
            'action' : {
                'function' : detectFacesDlibMmod,
                'args' : (),
                'kwargs' : {
                    'height' : dlib_mmod_detector_height,
                    'debug' : self.debug,
                    'verbose' : self.verbose
                }
            }
        }
    
    def mountAlignersDict(self):
        openface_aligner_model = rospy.get_param('/face_recognition/aligner/openface/model', 'shape_predictor_68_face_landmarks.dat')
        openface_aligner_image_dimension = rospy.get_param('/face_recognition/aligner/openface/image_dimension', 96)

        self.aligners_dict['openface'] = {
            'load' : {
                'function' : loadOpenfaceAlignerModel,
                'args' : (self.models_dir,),
                'kwargs' : {
                    'model' : openface_aligner_model,
                    'debug' : self.debug   
                }
            },
            'action' : {
                'function' : alignFaceOpenfaceAligner,
                'args' : (),
                'kwargs' : {
                    'image_dimension' : openface_aligner_image_dimension,
                    'debug' : self.debug,
                    'verbose' : self.verbose
                }    
            }
        }


    def mountEmbossersDict(self):

        openface_embosser_model = rospy.get_param('/face_recognition/embosser/openface/model', 'nn4.small2.v1.t7')
        openface_embosser_image_dimension = rospy.get_param('/face_recognition/embosser/openface/image_dimension', 96)
        openface_embosser_cuda = rospy.get_param('/face_recognition/embosser/openface/cuda', False)

        self.embossers_dict['openface'] = {
            'load' : {
                'function' : loadOpenfaceEmbosserModel,
                'args' : (self.models_dir,),
                'kwargs' : {
                    'model' : openface_embosser_model,
                    'image_dimension' : openface_embosser_image_dimension,
                    'cuda' : openface_embosser_cuda,
                    'debug' : self.debug 
                }
            },
            'action' : {
                'function' : extractFeaturesOpenfaceEmbosser,
                'args' : (),
                'kwargs' : {
                    'debug' : self.debug,
                    'verbose' : self.verbose
                }   
            }
        }


    def mountClassifiersDict(self):
        sklearn_classifier_model = rospy.get_param('/face_recognition/classifier/sklearn/model', 'classifier.pkl')
        sklearn_classifier_threshold = rospy.get_param('/face_recognition/classifier/sklearn/threshold', 0.5)

        self.classifiers_dict['sklearn'] = {
            'load' : {
                'function' : loadSklearnModel,
                'args' : (self.models_dir,),
                'kwargs' : {
                    'model' : sklearn_classifier_model,
                    'debug' : self.debug 
                }
            },
            'action' : {
                'function' : classifySklearn,
                'args' : (),
                'kwargs' : {
                    'threshold' : sklearn_classifier_threshold,
                    'debug' : self.debug,
                    'verbose' : self.verbose
                }
            }
        }


    def dlibRectangle2RosBoundingBox(self, rect):
        bounding_box = BoundingBox()
        if rect.tl_corner().x > 0:
            bounding_box.minX = rect.tl_corner().x
        if rect.tl_corner().y > 0: 
            bounding_box.minY = rect.tl_corner().y
        if rect.tl_corner().x + rect.width() > self.image_width:
            bounding_box.width = self.image_width - rect.tl_corner().x  
        else:    
            bounding_box.width = rect.width()
        if rect.tl_corner().y + rect.height() > self.image_height:
            bounding_box.height = self.image_height - rect.tl_corner().y  
        else:    
            bounding_box.height = rect.height()
        return bounding_box

    def numpyArray2RosVector(self, array):
        vector = array.tolist()
        return vector


    def loadDetector(self):
        detector_dict = self.detectors_dict[self.detector_model_id]['load']
        self.face_detector = detector_dict['function'](*detector_dict['args'], **detector_dict['kwargs'])

    def loadAligner(self):
        aligner_dict = self.aligners_dict[self.aligner_model_id]['load']
        self.face_aligner = aligner_dict['function'](*aligner_dict['args'], **aligner_dict['kwargs'])

    def loadEmbosser(self):
        embosser_dict = self.embossers_dict[self.embosser_model_id]['load']
        self.face_embosser = embosser_dict['function'](*embosser_dict['args'], **embosser_dict['kwargs'])

    def loadClassifier(self, model=''):
        if(model != ''):
            rospy.set_param('/face_recognition/classifier/lib/{}/model'.format(model), model)
            self.classifiers_dict[self.classifier_model_id]['load']['kwargs']['model'] = model
        classifier_dict = self.classifiers_dict[self.classifier_model_id]['load']
        self.face_classifier = classifier_dict['function'](*classifier_dict['args'], **classifier_dict['kwargs'])


    def detectFaces(self, bgr_image):
        detector_dict = self.detectors_dict[self.detector_model_id]['action']
        face_rects = detector_dict['function'](self.face_detector, bgr_image, *detector_dict['args'], **detector_dict['kwargs'])
        return face_rects

    def detectLargestFace(self, bgr_image):
        faces = self.detectFaces(bgr_image)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None
    
    def alignFace(self, bgr_image, face_rect):
        aligner_dict = self.aligners_dict[self.aligner_model_id]['action']
        aligned_face = aligner_dict['function'](self.face_aligner, bgr_image, face_rect, *aligner_dict['args'], **aligner_dict['kwargs'])
        return aligned_face

    def extractFeatures(self, aligned_face):
        embosser_dict = self.embossers_dict[self.embosser_model_id]['action']
        features_array = embosser_dict['function'](self.face_embosser, aligned_face, *embosser_dict['args'], **embosser_dict['kwargs']) 
        return features_array

    def classify(self, features_array):
        classifier_dict = self.classifiers_dict[self.classifier_model_id]['action']
        classification = classifier_dict['function'](self.face_classifier, features_array, *classifier_dict['args'], **classifier_dict['kwargs'])
        return classification

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
            rect = self.detectLargestFace(image.getRGB())
            aligned_face = self.alignFace(image.getRGB(), rect)
            bgr_image = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
            print(os.path.join(raw_aligned_dir, image.cls, image.name + '.jpg'))
            cv2.imwrite(os.path.join(raw_aligned_dir, image.cls, image.name + '.jpg'), bgr_image)

    def extractDatasetFeatures(self):
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
            features = self.extractFeatures(image.getRGB())
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

        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        labels_num = label_encoder.transform(labels)
        num_classes = len(label_encoder.classes_)

        if classifier_type == 'lsvm':
            classifier = SVC(C=1, kernel='linear', probability=True)
        elif classifier_type == 'rsvm':
            classifier = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif classifier_type == 'gssvm':
            param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
            ]
            classifier = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif classifier_type == 'gmm':
            classifier = GMM(n_components=num_classes)
        elif classifier_type == 'dt':
            classifier = DecisionTreeClassifier(max_depth=20)
        elif classifier_type == 'gnb':
            classifier = GaussianNB()
        elif classifier_type == 'knn':
            classifier = KNeighborsClassifier(n_neighbors=10)
        else:
            return False

        classifier.fit(embeddings, labels_num)
        fName = self.models_dir + '/sklearn/' + classifier_name
        with open(fName, 'w') as f:
            pickle.dump((label_encoder, classifier), f)
        return True  


    def recognitionProcess(self, ros_msg):
        rospy.loginfo('Image ID: {}'.format(ros_msg.header.seq))
        bgr_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="bgr8")

        self.image_height = bgr_image.shape[0]
        self.image_width = bgr_image.shape[1]

        face_rects = self.detectFaces(bgr_image)
        if len(face_rects) == 0:
            rospy.loginfo("Recognition FPS: {:.2f} Hz.".format((1/(rospy.get_rostime().to_sec() - self.last_recognition))))
            self.last_recognition = rospy.get_rostime().to_sec()
            return None

        faces_description = []
        for face_rect in face_rects:
            face_description = Description()

            aligned_face = self.alignFace(bgr_image, face_rect)

            features_array = self.extractFeatures(aligned_face)

            classification = self.classify(features_array)

            label_class, confidence = classification
            
            bounding_box = self.dlibRectangle2RosBoundingBox(face_rect)
            features = self.numpyArray2RosVector(features_array)

            face_description.label_class = label_class
            face_description.probability = confidence
            face_description.bounding_box = bounding_box
            faces_description.append(face_description)

        recognized_faces = Recognitions()
        recognized_faces.image_header = ros_msg.header

        recognized_faces.header.stamp = rospy.get_rostime()
        recognized_faces.header.frame_id = "face_recognition"

        recognized_faces.descriptions = faces_description

        rospy.loginfo("Recognition FPS: {:.2f} Hz.".format((1/(rospy.get_rostime().to_sec() - self.last_recognition))))
        self.last_recognition = rospy.get_rostime().to_sec()

        return recognized_faces

    def trainingProcess(self, ros_srv):
        self.alignDataset()
        self.extractDatasetFeatures()
        ans = self.trainClassifier(ros_srv.classifier_type, ros_srv.classifier_name)
        return ans 

