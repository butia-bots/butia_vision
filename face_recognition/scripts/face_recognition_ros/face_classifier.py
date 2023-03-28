#! /usr/bin/env python

import pickle
from face_recognition_ros.decorators import *
import os
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture as GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier

@load(model_id='sklearn')
@debug
def loadSklearnModel(models_dir, model='classifier.pkl', debug=False):
    try:
        with open(os.path.join(models_dir, 'sklearn', model), 'rb') as model_file:
            (cl_label, classifier_model) = pickle.load(model_file, encoding='latin1')
            return (cl_label, classifier_model)
    except FileNotFoundError:
        print('no classifier found')
        return (None)
@action(action_name='classify')
@debug
def classifySklearn(classifier, array, threshold=0.5, verbose=True, debug=False):
    if classifier is not None:
        cl_label, classifier_model = classifier
        rep = array.reshape(1, -1)
        predictions = classifier_model.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = cl_label.inverse_transform([maxI])[0]
        confidence = predictions[maxI]
        if confidence > threshold:
            return (person, confidence)
        else:
            return ('anknow', confidence)
    else:
        confidence = 1
        return ('anknow', confidence)


