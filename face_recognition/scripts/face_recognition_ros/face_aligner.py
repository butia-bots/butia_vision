#! /usr/bin/env python
import openface
import cv2
from decorators import *
import os

@load(model_id='Openface Aligner')
@debug
def loadOpenfaceAlignerModel(models_dir, model = 'shape_predictor_68_face_landmarks.dat', debug=False):
    openface_aligner = openface.AlignDlib(os.path.join(models_dir, 'openface', model))
    return openface_aligner

@action(action_name='align')
@debug
def alignFaceOpenfaceAligner(aligner, image, rect, image_dimension=96, verbose=True, debug=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    aligned_face = aligner.align(image_dimension, image_rgb, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return aligned_face
