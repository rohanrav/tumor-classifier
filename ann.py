#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:49:27 2019

@author: rohi
"""

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

# Importing the libraries required for the ANN
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dropout(0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(0.1))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Creating filepath variable
filepath = "weights.h5"
# Creating the ModelCheckpoint Object to save our best weights
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

# Fitting the ANN to the Training set
# Setting the ANN to use the checkpoint object defined earlier
classifier.fit(X_train, Y_train, batch_size = 16, epochs = 100, callbacks=[checkpoint])