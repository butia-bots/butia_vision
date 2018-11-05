#! /usr/bin/env python
from decorators import *
import openface
import os

@load(model_id='Openface')
@debug
def loadOpenfaceEmbosserModel(models_dir, model='nn4.small2.v1.t7', image_dimension=96, cuda=True, debug=False):
    facenet_embosser = openface.TorchNeuralNet(os.path.join(models_dir, 'openface', model), image_dimension, cuda=cuda)
    return facenet_embosser

@action(action_name='embbed')
@debug
def extractFeaturesOpenfaceEmbosser(embosser, image, verbose=True, debug=False):
    features = embosser.forward(image)
    return features