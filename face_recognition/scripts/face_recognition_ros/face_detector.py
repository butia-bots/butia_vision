#! /usr/bin/env python
import cv2
import openface
from decorators import *
import os

from dlib import rectangle, rectangles

def numpyndArray2dlibRectangles(array):
    rects = rectangles()
    for (x, y, width, height) in array:
        rects.append(rectangle(x, y, x + width, y + height))
    return rects


@load(lib_name='OpenCV')
@debug
def loadOpencvModels(models_dir, model = 'haarcascade_frontalface_alt.xml', cuda = True, debug=False):
    if(cuda):
        model = 'cuda/' + model
    opencv_detector = cv2.CascadeClassifier(os.path.join(models_dir, 'opencv', model))
    return opencv_detector

@load(lib_name='dlib')
@debug
def loadDlibModels(models_dir, model = 'default', debug=False):
    dlib_aligner = openface.AlignDlib(os.path.join(models_dir, 'dlib', model))
    return dlib_aligner

@action(action_name='detection')
@debug
def detectFacesOpencv(detector, image, scale_factor=1.3, min_neighbors=5, verbose=True, debug=False):
        faces = detector.detectMultiScale(image, scale_factor, min_neighbors)
        faces = numpyndArray2dlibRectangles(faces)
        return faces

@action(action_name='detection')
@debug
def detectFacesDlib(detector, image, verbose=True, debug=False):
    faces = detector.getAllFaceBoundingBoxes(image)
    return faces

@action(action_name='align')
@debug
def alignFaceDlib(aligner, image, rect, image_dimension=96, verbose=True, debug=False):
    aligned_face = aligner.align(image_dimension, image, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return aligned_face