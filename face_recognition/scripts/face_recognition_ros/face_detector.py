#! /usr/bin/env python
import cv2
import openface
import decorators

from dlib import rectangles

def numpyndArray2dlibRectangles(self, array):
    rects = rectangles()
    for (x, y, width, height) in array:
        rects.append(rectangle(x, y, x + width, y + height))
    return rects


@load(lib_name='OpenCV')
@debug
def loadOpencvModels(models_dir, model = 'haarcascade_frontalface_alt.xml', cuda = True):
    if(cuda):
        opencv_model = 'cuda/' + opencv_model
    opencv_detector = cv2.CascadeClassifier(os.path.join(models_dir, 'opencv', opencv_model))
    return opencv_detector

@load(lib_name='dlib')
@debug
def loadDlibModels(models_dir, model = 'shape_predictor_68_face_landmarks.dat'):
    dlib_aligner = openface.AlignDlib(os.path.join(models_dir, 'dlib', dlib_model))
    return dlib_aligner

@action(action_name='detection')
@debug
def detectFacesOpencv(detector, image, scale_factor=1.3, min_neighbors=5):
        faces = detector.detectMultiScale(image, scale_factor, min_neighbors)
        faces = numpyndArray2dlibRectangles(faces)
        return faces

@action(action_name='detection')
@debug
def detectFacesDlib(detector, image):
    faces = detector.getAllFaceBoundingBoxes(image)
    return faces

@action(action_name='align')
@debug
 def alignFaceDlib(aligner, image, rect, image_dimension=96):
    aligned_face = aligner.align(image_dimension, image, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)







    

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

    

   

    