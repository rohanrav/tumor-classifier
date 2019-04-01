#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:49:27 2019

@author: rohi
"""
# Importing the libraries nessecary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
dataset = pd.read_csv('cancer.csv')
X = dataset.iloc[:, 2:32].values # Seperating Independent Variables into X
Y = pd.DataFrame(dataset.iloc[:, 1].values) # Seperating Dependent Variables into Y Pandas Dataframe

# Encoding the Dependent Variables 
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into training data and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Importing the load_model function required to load our model weights
from keras.models import load_model
filepath = "weights.h5"
classifier = load_model(filepath)

y_pred = classifier.predict(X_test) # Using the classifier to predict testing data
# .predict returns a percentage (sigmoid)
y_pred = (y_pred > 0.5) # Must have binary results, setting outcomes of < 0.5 to be 1, and outcomes <= 0.5 to be 0

from sklearn.metrics import accuracy_score  # Getting how accurate our predicted test results were 
print("The predicted results for our test set had an accuracy of: ", (accuracy_score(Y_test, y_pred)*100), "%.")
