#! /usr/bin/env python
import pickle
from decorators import *
import os
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

@load(model_id='sklearn')
@debug
def loadSklearnModel(models_dir, model='classifier.pkl', debug=False):
    with open(os.path.join(models_dir, 'sklearn', model), 'rb') as model_file:
        (cl_label, classifier_model) = pickle.load(model_file)
        print cl_label.classes_
        return (cl_label, classifier_model)

@action(action_name='classify')
@debug
def classifySklearn(classifier, array, threshold=0.5, verbose=True, debug=False):
    cl_label, classifier_model = classifier
    rep = array.reshape(1, -1)
    predictions = classifier_model.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = cl_label.inverse_transform(maxI)
    confidence = predictions[maxI]
    if confidence > threshold:
        return (person.decode('utf-8'), confidence)
    else:
        return ('unknow', confidence)