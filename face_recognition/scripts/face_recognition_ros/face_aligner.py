#! /usr/bin/env python
import openface
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
    aligned_face = aligner.align(image_dimension, image, rect, landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return aligned_face
