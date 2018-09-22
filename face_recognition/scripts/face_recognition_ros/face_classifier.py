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

@load(lib_name='sklearn')
@debug
def loadSklearnModels(models_dir, model='classifier.pkl', debug=False):
    with open(os.path.join(models_dir, 'classifier', model), 'rb') as model_file:
        (cl_label, classifier_model) = pickle.load(model_file)
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
        return ('Unknow', confidence)

@debug
def trainClassifier(classifier_type, classifier_name, feature_data, debug=False):
        #features_dir = os.path.join(self.dataset_dir, 'features')
        #features_file = open(os.path.join(features_dir, 'features.json'), 'rw')
        #features_data = json.load(features_file)

        labels = []
        embeddings = []
        for key in features_data.keys():
            labels += len(features_data[key].keys()) * [key]
            embeddings += features_data[key].values()

        embeddings = np.array(embeddings)

        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        labels_num = label_encoder.transform(labels)
        num_classes = len(label_encoder.classes_)

        #tentar transformar essa cadeia de if's em um dicionario, como eh feito no FaceDetector
        if classifier_type == 'lsvm':
            classifier = SVC(C=1, kernel='linear', probability=True)
        elif classifier_type == 'rsvm':
            classifier = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif classifier_type == 'gssvm':
            param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
            ]
            classifier = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif classifier_type == 'gmm':
            classifier = GMM(n_components=num_classes)
        elif classifier_type == 'dt':
            classifier = DecisionTreeClassifier(max_depth=20)
        elif classifier_type == 'gnb':
            classifier = GaussianNB()
        else:
            return False

        classifier.fit(embeddings, labels_num)
        fName = self.models_dir + '/classifier/' + classifier_name
        with open(fName, 'w') as f:
            pickle.dump((label_encoder, classifier), f)
        return True  