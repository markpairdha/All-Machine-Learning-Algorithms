# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 13:07:23 2019

@author: markp
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read datasets
dataset = pd.read_csv('Data.csv')
x =dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values

#handing missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])

#handling categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#dividing the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#feature scaling for applying mathematics
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)