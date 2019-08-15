#! /usr/bin/env python

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, clone
import numpy as np

class TwoFac_Classifier(BaseEstimator):
    def __init__(self, multi_class, one_class, classes):
        self.multi_class = multi_class
        self.one_class_ex = one_class
        self.classes = classes
        self.classifiers_dict = {}
    
    def fit(self, samples, labels):
        self.multi_class.fit(samples,labels)
        samples_dict = {}
        for i, num in enumerate(labels):
            if num not in samples_dict:
                samples_dict[num] = []
            samples_dict[num] += [(samples[i]).tolist()]
        for class_ in range(self.classes):
            sample = np.array(samples_dict[class_])
            self.classifiers_dict[class_] = clone(self.one_class_ex)
            self.classifiers_dict[class_].fit(sample, [1] * len(samples_dict[class_]))
    
    def predict_proba(self, rep):
        response = self.multi_class.predict_proba(rep)
        predictions = response.ravel()
        maxI = np.argmax(predictions)
        prediction = self.classifiers_dict[maxI].predict(rep)
        if (prediction[0] == 1):
            for i in range(len(predictions)):
                if(i!= maxI):
                    predictions[i] = 0
            return np.array(predictions)
        else:
            response = [0]*len(response[0])
            return np.array(response)
        
