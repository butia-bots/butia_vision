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
def detectFacesOpencvDnn(detector, image, threshold=0.7, scale_factor=1.0, height=300, mean=[104, 117, 123], verbose=True, debug=False):
    image_height = image.shape[0]
    image_width = image.shape[1]
    width = 0
    if height != image_height:
        relation = float(image_width) / image_height
        width = int(relation * height)
    else:
        width = image_width
    size = (width, height)
    blob = cv2.dnn.blobFromImage(image, scale_factor, size, mean)

    detector.setInput(blob)
    detections = detector.forward()
    faces = rectangles()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= threshold:
            x1 = int(detections[0, 0, i, 3] * image_width)
            y1 = int(detections[0, 0, i, 4] * image_height)
            x2 = int(detections[0, 0, i, 5] * image_width)
            y2 = int(detections[0, 0, i, 6] * image_height)
            faces.append(rectangle(x1, y1, x2, y2))
    return faces

@action(action_name='detection')
@debug
def detectFacesDlibHog(detector, image, height=300, verbose=True, debug=False):
    image_height = image.shape[0]
    image_width = image.shape[1]
    width = 0
    if height != image_height:
        relation = float(image_width) / image_height
        width = int(relation * height)
    else:
        width = image_width

    scale_height = float(image_height) / height
    scale_width = float(image_width) / width

    image_small = cv2.resize(image, (width, height))

    image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
    faces_small = detector(image_small, 0)
    faces = rectangles()
    for face in faces_small:
        faces.append(rectangle(int(face.left()*scale_width), int(face.top()*scale_height),
                  int(face.right()*scale_width), int(face.bottom()*scale_height)))
    return faces

@action(action_name='detection')
@debug
def detectFacesDlibMmod(detector, image, height=300, verbose=True, debug=False):
    image_height = image.shape[0]
    image_width = image.shape[1]
    width = 0
    if height != image_height:
        relation = float(image_width) / image_height
        width = int(relation * height)
    else:
        width = image_width

    scale_height = float(image_height) / height
    scale_width = float(image_width) / width

    image_small = cv2.resize(image, (width, height))

    image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
    faces_small = detector(image_small, 0)
    faces = rectangles()
    for face in faces_small:
        faces.append(rectangle(int(face.rect.left()*scale_width), int(face.rect.top()*scale_height),
                  int(face.rect.right()*scale_width), int(face.rect.bottom()*scale_height)))
    return faces