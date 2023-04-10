"""
Created on Mon Jun  8 11:31:30 2020

@author: Ilyas IRGUI -----------> IA&GI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

Y_train = y_train.T
Y_test = y_test.T

X_train = np.hstack((X_train, np.ones((Y_train.shape[0],1))))
X_test = np.hstack((X_test, np.ones((Y_test.shape[0],1))))

X_train = X_train.T
X_test = X_test.T

# =============================================================================
# print(X_test.shape)
# print(X_train.shape)
# print(Y_train.shape)
# print(Y_test.shape)
# =============================================================================

# =============================================================================
# ---------- Initial Weight ------------#
# =============================================================================

teta = np.full((X_train.shape[0],1),0.05)

# print(teta.shape)

# =============================================================================
# #---------- Sigmoid Function ---------#
# =============================================================================

def sigmoid(z):
 	return 1.0 / (1.0 + np.exp(-z))

# =============================================================================
# #-------- Cost Function ---------------#
# =============================================================================

def cost(X_train, teta, Y_train):
    h = sigmoid(np.dot(teta.T, X_train))
    loss = Y_train * np.log(h) + (1 - Y_train) * np.log(1 - h)
    cost = -1 / X_train.shape[1] * np.sum(loss)
    return cost

# =============================================================================
# #------------- Gradient --------------#
# =============================================================================

def grad(X_train, teta, Y_train):
    h = sigmoid(np.dot(teta.T, X_train))
    mu = np.dot(X_train, (Y_train - h).T)
    grad = -1/X_train.shape[1] * mu
    return grad

# =============================================================================
# #-------- Gradiant Descent -------#
# =============================================================================

def GD(X, teta, Y, nb_iterations, learning_rate):

    cost_history = []
    index = []
    for i in range(nb_iterations):
        teta = teta - learning_rate * grad(X, teta, Y)
        if i % 10 == 0:
            cost_history.append(cost(X, teta, Y))
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost(X, teta, Y)))

    plt.plot(index,cost_history)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return teta,cost_history

# =============================================================================
# #-------- Prediction -------#
# =============================================================================

def predict(teta_final,x_test):
    h = sigmoid(np.dot(teta_final.T,x_test))
    Y_prediction = np.ones((1,x_test.shape[1]))
    for i in range(h.shape[1]):
        if h[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction




teta_final, cost_history = GD(X_train, teta, Y_train, 100, 0.1)


# =============================================================================
# #-------- GD Accuracy Test -------#
# =============================================================================

Y_prediction = predict(teta_final,X_test).T
print("-----------------------------")
print("GD Accuracy Test: {0:.5%}".format(accuracy_score(Y_prediction, Y_test)))
print("-----------------------------")
