# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:27:26 2020

@author: hp
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =============================================================================
# #----------Data Cleaning & Preprocessing---------#
# =============================================================================

data = pd.read_csv("datasets_180_408_data.csv")

# data.info()

dataa = data.iloc[:, 2:-1]

# print(dataa.head())

x = dataa
y = data['diagnosis']

# print(y.values)

# print(x)

y = preprocessing.LabelEncoder().fit_transform(y)
# print(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

sc = StandardScaler ()
X_train = sc.fit_transform (X_train)
X_test = sc.transform (X_test)

# print(X_train)
# print(X_test

Y_train = y_train
Y_test = y_test

X_train = np.hstack((X_train, np.ones((Y_train.shape[0],1))))
X_test = np.hstack((X_test, np.ones((Y_test.shape[0],1))))


Y_train=Y_train.reshape(-1,1)
Y_test=Y_test.reshape(-1,1)

# =============================================================================
# print(X_test.shape)
# print(X_train.shape)
# print(Y_train.shape)
# print(Y_test.shape)
# =============================================================================

# =============================================================================
# ---------- Initial Weight ------------#
# =============================================================================

teta = np.zeros((X_train.shape[1],1))

# print(teta.shape)

# =============================================================================
# #---------- Sigmoid Function ---------#
# =============================================================================

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

# =============================================================================
# #-------- Cost Function ---------------#
# =============================================================================

def cost_function(X,y,theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return -(1 / m) * np.sum(np.multiply(y,np.log(h)) + np.multiply((1 - y),np.log(1 - h)))

# =============================================================================
# #------------- Gradient --------------#
# =============================================================================

def grad(X,y,theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return ((1 / m) * np.dot(X.T,(h - y)))


# =============================================================================
# #------------- Stochastic Gradient Descent --------------#
# =============================================================================

def minibatch_gradient_descent(X,y,teta,learning_rate,n_iterations,batch_size):
    m = len(y)
    cost_history = []
    for i in range(n_iterations):
       indices=np.random.permutation(m)
       X=X[indices]
       y=y[indices]
       for i in range(0,m,batch_size):
           X_i=X[i:i+batch_size]
           y_i=y[i:i+batch_size]
           teta = teta - learning_rate * grad(X_i,y_i,teta)
       cost_history.append(cost_function(X_i,y_i,teta))
    return teta,cost_history


# =============================================================================
# #-------- Prediction -------#
# =============================================================================

def predict(teta_final,x_test):
    h = sigmoid(x_test.dot(teta_final))
    Y_prediction = np.ones((1,x_test.shape[0]))
    for i in range(h.shape[0]):
        if h[i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction




theta_final,cost_history = minibatch_gradient_descent(X_train, Y_train, teta, 0.01, 100, 50)

# =============================================================================
# #-------- SGD Accuracy Test -------#
# =============================================================================

Y_prediction = predict(theta_final,X_test).T
print("-----------------------------")
print("SGD Accuracy Test: {0:.3%}".format(accuracy_score(Y_prediction, Y_test)))
print("-----------------------------")











