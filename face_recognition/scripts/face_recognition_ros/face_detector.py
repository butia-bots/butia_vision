#! /usr/bin/env python
import cv2
import openface

from dlib import rectangles

class FaceDetector():
    def __init__(self, detector_lib = 'opencv', aligner_lib = 'dlib'):
        self.detector_lib = detector_lib
        self.aligner_lib = aligner_lib

        self.detectors_dict = {}
        self.aligners_dict = {}

    def loadOpencvModels(self, opencv_model = 'haarcascade_frontalface_alt.xml', opencv_cuda = True, (scale_factor = 1.3, min_neighbors = 5)):
        self.opencv_args = (scale_factor, min_neighbors)
        if(opencv_cuda):
            opencv_model = 'cuda/' + opencv_model
        self.opencv_detector = cv2.CascadeClassifier(os.path.join(self.models_dir, 'opencv', opencv_model))
        self.detectors_dict['opencv'] = detectFacesOpencv

    def loadDlibModels(self, dlib_model = 'shape_predictor_68_face_landmarks.dat', image_dimension = 96):
        self.image_dimension = image_dimension
        self.dlib_aligner = openface.AlignDlib(os.path.join(self.models_dir, 'dlib', dlib_model))
        self.detectors_dict['dlib'] = detectFacesDlib
        self.aligners_dict['dlib'] = alignFaceDlib

    def detectFacesOpencv(self, image):
        faces = self.opencv_detector.detectMultiScale(image, self.opencv_args*)
        faces = self.numpyndArray2dlibRectangles(faces)
        return faces

    def detectFacesDlib(self, image):
        faces = self.dlib_aligner.getAllFaceBoundingBoxes(image)
        return faces

    def detectFaces(self, image):
        #now_s = rospy.get_rostime().to_sec()
        try:
            faces = self.detectors_dict[self.detector_lib](image)
        except KeyError:
            print(self.detector_lib + ' model is not loaded.')
            return None
        
        if len(faces)>0:
            return faces
        else:
            return None
    
    def detectLargestFace(self, image):
        #now_s = rospy.get_rostime().to_sec()
        faces = self.detectFaces(self, image)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None
        #rospy.loginfo("Face detection took: " + str(rospy.get_rostime().to_sec() - now_s) + " seconds.")

    def alignFaceDlib(self, image, rect):
        #now_s = rospy.get_rostime().to_sec()
        aligned_face = self.dlib_aligner.align(self.image_dimension, image, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
        #rospy.loginfo("Face aligner took: " + str(rospy.get_rostime().to_sec() - now_s) + " seconds.")
        return aligned_face

    def alignFace(self, image, rect):
        try:
            aligned_face = self.aligners_dict[self.aligner_lib](image, rect)
        except KeyError:
            print(self.aligner_lib + ' model is not loaded.')
            return None
        return aligned_face

    def numpyndArray2dlibRectangles(self, array):
        rects = rectangles()
        for (x, y, width, height) in array:
            rects.append(rectangle(x, y, x + width, y + height))
        return rects