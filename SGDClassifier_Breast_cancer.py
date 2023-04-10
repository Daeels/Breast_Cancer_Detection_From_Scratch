# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:47:29 2020

@author: hp
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("datasets_180_408_data.csv")

di = data.iloc[:, 2:-1]

x = di
y = data['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

sc = StandardScaler ()
X_train = sc.fit_transform (X_train)
X_test = sc.transform (X_test)

Y_train = preprocessing.LabelEncoder().fit_transform(y_train)
y_test = preprocessing.LabelEncoder().fit_transform(y_test)

clf = SGDClassifier(random_state=42, loss='log')
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)



print('--------------------------------')
print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print('--------------------------------')
