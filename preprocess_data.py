import pandas as pd
import numpy as np
import tensorflow as tf



def get_data():
    view_data = pd.read_csv('data.csv', header=None)
    data = view_data.as_matrix()
    Y = data[:,-1]
    X = data[:,:-1]
    
    N, D = X.shape
    M = 5
    
    K = len(set(Y))
    T = np.zeros((N, 3))
    
    # prepare for one-hot encoding
    setosa = "Iris-setosa"
    versicolor = "Iris-versicolor"
    virginica = "Iris-virginica"
    # one hot encoding for targets
    for i in range(len(Y)):
        if Y[i] == setosa:
            T[i][0] = 1
        elif Y[i] == versicolor:
            T[i][1] = 1
        elif Y[i] == virginica:
            T[i][2] = 1
    
    # normalize features
    X, means, stds = normalize_input(X)
   
        
    Y_argmax = np.argmax(T, 1)       
    return X.astype(float), T, Y_argmax, N, D, K, M, means, stds


def normalize_input(X):
    means = np.zeros(len(X[0]))
    stds = np.zeros(len(X[0]))
    for i in range(len(X[0])):
        means[i] = X[:,i].mean()
        stds[i] = X[:,i].std()
        X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()
    return X, means, stds
    


