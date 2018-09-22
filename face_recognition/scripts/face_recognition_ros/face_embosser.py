#! /usr/bin/env python
from decorators import *
import openface
import os

@load(lib_name='FaceNet')
def loadFacenetModels(models_dir, model='nn4.small2.v1.t7', image_dimension=96, cuda=True, debug=False):
    facenet_embosser = openface.TorchNeuralNet(os.path.join(models_dir, 'openface', model), image_dimension, cuda=cuda)
    return facenet_embosser

@action(action_name='embbed')
@debug
def extractFeaturesFacenet(embosser, image, verbose=True, debug=False):
    features = embosser.forward(image)
    return features
'''
@debug
def generateDatasetFeatures(dataset_dir):
    raw_aligned_dir = os.path.join(dataset_dir, 'raw_aligned')
    features_dir = os.path.join(dataset_dir, 'features')
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
        features = extractFeaturesFromImage(image.getRGB())
        features_data[label][name] = numpyArray2RosVector(features)

    json.dump(features_data, features_file)
'''