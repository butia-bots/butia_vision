#! /usr/bin/env python
import cv2
import dlib
from decorators import *
import os

from dlib import rectangle, rectangles

def numpyndArray2dlibRectangles(array):
    rects = rectangles()
    for (x, y, width, height) in array:
        rects.append(rectangle(x, y, x + width, y + height))
    return rects


@load(model_id='OpenCV Cascade')
@debug
def loadOpencvCascadeModel(models_dir, model = 'haarcascade_frontalface_alt.xml', cuda = True, debug=False):
    if(cuda):
        model = 'cuda/' + model
    opencv_detector = cv2.CascadeClassifier(os.path.join(models_dir, 'opencv', model))
    return opencv_detector

@load(model_id='OpenCV DNN')
@debug
def loadOpencvDnnModel(models_dir, model = 'tensorflow', debug = False):
    opencv_detector = None
    if model == 'caffe':
        model_file = os.path.join(models_dir, 'opencv', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
        config_file = os.path.join(models_dir, 'opencv', 'deploy.prototxt')
        opencv_detector = cv2.dnn.readNetFromCaffe(config_file, model_file)
    elif model == 'tensorflow':
        model_file = os.path.join(models_dir, 'opencv', 'opencv_face_detector_uint8.pb')
        config_file = os.path.join(models_dir, 'opencv', 'opencv_face_detector.pbtxt')
        opencv_detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    return opencv_detector

@load(model_id='Dlib HOG')
@debug
def loadDlibHogModel(models_dir, model = 'default', debug=False):
    dlib_detector = None
    if(model == 'default'):
        dlib_detector = dlib.get_frontal_face_detector()
    return dlib_detector

@load(model_id='Dlib MMOD')
@debug
def loadDlibMmodModel(models_dir, model = 'mmod_human_face_detector.dat', debug=False):
    dlib_detector = dlib.cnn_face_detection_model_v1(os.path.join(models_dir, 'dlib', model))
    return dlib_detector

@action(action_name='detection')
@debug
def detectFacesOpencvCascade(detector, image, scale_factor=1.3, min_neighbors=5, verbose=True, debug=False):
    faces = detector.detectMultiScale(image, scale_factor, min_neighbors)
    faces = numpyndArray2dlibRectangles(faces)
    return faces

@action(action_name='detection')
@debug
def detectFacesOpencvDnn(detector, image, threshold = 0.7, verbose=True, debug=False):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

@action(action_name='detection')
@debug
def detectFacesDlibHog(detector, image, verbose=True, debug=False):
    faces = detector.getAllFaceBoundingBoxes(image)
    return faces

@action(action_name='detection')
@debug
def detectFacesDlibMmod(detector, image, verbose=True, debug=False):
    faces = detector.getAllFaceBoundingBoxes(image)
    return faces