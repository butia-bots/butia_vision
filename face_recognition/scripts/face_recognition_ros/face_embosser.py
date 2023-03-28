#! /usr/bin/env python
from face_recognition_ros.decorators import *
import openface
import os
import torch
import numpy as np
from keras.utils import img_to_array, load_img

from keras.applications.imagenet_utils import preprocess_input
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

@load(model_id='vgg_face')
@debug
def loadVGGFaceModel(models_dir, model = 'VGG_FACE.t7' , cuda=True, debug=False):
    vgg_model = openface.TorchNeuralNet(os.path.join(models_dir, 'vggface', model), cuda=cuda)
    return vgg_model

@action(action_name='embbed')
@debug
def extractFeaturesVGGFace(vgg_model, image, verbose=True, debug=False):
    
    img = load_img(image, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    input_tensor = torch.from_numpy(img).float()
    features = vgg_model.forward(input_tensor)
    return features