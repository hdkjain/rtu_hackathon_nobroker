#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 05:06:05 2018

@author: crazyconda
"""

#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler


#import datasert
dataset = pd.read_csv('newdata.csv')
X = dataset.iloc[:,[1,7,8,9,13,15,19,23,32]].values
features = pd.DataFrame(X)
Y = dataset.iloc[:, -1].values
labels = pd.DataFrame(Y)


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# LabelEncoder
le = LabelEncoder()
features.values[:,0] = le.fit_transform(features.values[:,0])

#OneHotEncoding the labelled data
onehotencoder = OneHotEncoder(categorical_features=[0])
features = onehotencoder.fit_transform(features).toarray()

#Splitting the dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


#Fitting Linear Regression to the dataset
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0)
classifier.fit(features_train, labels_train)

#predicting the test set result
labels_pred = classifier.predict(features_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

#score
classifier.score = classifier.score(features, labels)

#visualize
from matplotlib.color import ListedColormap
features_set, labels_set = features_train, labels_train
features1, features2 = np.meshgrid(np.arange(start = features_set[:,0].min() -1, stop = features_set[:,0].max() +1, step = 0.01))

