#! /usr/bin/env python

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator, clone
import numpy as np

class twofac_classifier(BaseEstimator):
    def __init__(self, multi_class, one_class, classes):
        self.multi_class = multi_class
        self.one_class_ex = one_class
        self.classes = classes
        self.classifiers_dict = {}
    
    def fit(self, samples, labels):
        self.multi_class.fit(samples,labels)
        samples_dict = {}
        for class_ in range(self.classes):
            self.classifiers_dict[class_] = clone(self.one_class_ex)
            self.classifiers_dict[class_].fit((samples_dict[class_]).reshape(-1,1), [1] * len(samples_dict[class_]))
    
    def predict_proba(self, rep):
        response = self.multi_class.predict_proba(rep)
        predictions = response.ravel()
        maxI = np.argmax(predictions)
        confidence = predictions[maxI]
        
        if (self.classifiers_dict[maxI].predict_(rep) == [1]):
            for i in range(len(response)):
                if(i!= maxI):
                    response[i] = 0
        
            return response
        else:
            return ([0]*len(response))
        
        #op1
        if confidence > 0.5:
            belong_class = self.one_class[maxI].predict(rep)
            if
        else:
            return response
            belong_class = self.one_class[maxI].predict(rep)
            if belong_class[0] == 0:
                return response
            else:
               
    # retornar array com confiança ou, se for pro one class, retornar 0
        

