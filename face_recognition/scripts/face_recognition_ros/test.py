import os
import json
import rospkg
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import svm
from sklearn.grid_search import GridSearchCV

PACK_DIR = rospkg.RosPack().get_path('face_recognition')

dataset_dir = os.path.join(PACK_DIR, 'dataset')
features_dir = os.path.join(dataset_dir, 'features')
features_file = open(os.path.join(features_dir, 'features.json'), 'rw')
features_data = json.load(features_file)

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

samples_dict = {}
for class_ in range(num_classes):
    samples_dict[class_] = embeddings[class_]


param_grid = {'nu': [0.1],
              'gamma': [0.001, 0.0001, 0.00001 , "auto"],
              'kernel': ['rbf'],
              'tol' : [0.001]}
score = 'precision'
one_class = GridSearchCV(svm.OneClassSVM(), param_grid, cv=6,
                         scoring='%s_macro' % score)

one_class.fit((samples_dict[0]).reshape(-1,1), [1] * len(samples_dict[0]))

print(one_class.predict(samples_dict[2][2]))

