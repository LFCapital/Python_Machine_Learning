import numpy as np
import pandas as pd
from datetime import datetime
'''

This code shows how to build a linear regression model with Gradient Descent and Normal Equation method.
Generally speaking, if the dataset size is small we can use Normal Equation method,
otherwise we usually use gradient descent or stocastic gradient descent.

The training data comes from Andrew Ng Machine Learning courses at Coursera homework 1.
'''

def computeCost(X,Y,w):
    n = X.shape[0]
    residual_squre_list = [(X[i,:].dot(w)-Y[i,:])**2 for i in xrange(n)]
    return sum(residual_squre_list)/float(2*n)

def GradientDescent(X,Y,w,eta,iterations):
    # this is batch gradient descent    
    n = X.shape[0]    
    for i in xrange(iterations):
        tempresult = np.zeros((w.shape[0],1))
        # w_j = w_j - eta/m*sum_{i=1}^{m}((w.T*x(i)-y(i))*x(i,j))
        for j in xrange(w.shape[0]):
           sumlist = sum([(X[k,:].dot(w) - Y[k,:]) * X[k,j] for k in xrange(n)])
           tempresult[j,:] = eta * sumlist / float(n) 
        # update w simultaneously
        w = w - tempresult
    return w

def GradientDescentVector(X,Y,w,eta,iterations):
    # this is Vectorized gradient descent method
    n = X.shape[0]
    for i in xrange(iterations):
        inner_sum = X.T.dot(X.dot(w) - Y)
        w = w - eta * inner_sum / float(n) 
    return w

def normalEqn(X,Y):
    # w = (X^T*X)^-1*X^T*Y and normal equation doesn't need feature normalize
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def featureNormalize(X):
    X_norm = X.astype(np.float64)
    for i in xrange(X.shape[1]):
        X_norm[:,i] = X_norm[:,i] - np.mean(X[:,i])
        X_norm[:,i] = X_norm[:,i] / np.std(X[:,i])
    return X_norm

def OneVariable_LinearRegression():
    train = pd.read_csv("D://ex1data1.txt",header = None, sep = ',')
    train = np.array(train)  
    X = train[:,0:1]
    Y = train[:,1:]
    # add x_0
    X = np.hstack((np.ones((X.shape[0],1)),X))
    # gradient descent settings
    iterations = 1500
    eta = 0.01
    w = np.zeros((2,1))
    # print the origin cost before gradient descent
    print computeCost(X,Y,w)
    w = GradientDescent(X,Y,w,eta,iterations)
    print computeCost(X,Y,w)
    # from the two cost, we can see that gradient descent indeed minimize the objective function    

def MultipleVariable_LinearRegression():
    train = pd.read_csv("D://ex1data2.txt",header = None, sep = ',')
    train = np.array(train)  
    X = train[:,0:2]
    Y = train[:,2:]
    
    # feature scaling
    X = featureNormalize(X)
    # add x_0
    X = np.hstack((np.ones((X.shape[0],1)),X))
    eta = 0.1
    iterations = 50
    w = np.zeros((X.shape[1],1))
    # gradient descent via loop
    begin = datetime.now()
    w = GradientDescent(X,Y,w,eta,iterations)
    print w
    print datetime.now() - begin
    # Vectorized gradient descent
    begin = datetime.now()
    w = GradientDescentVector(X,Y,w,eta,iterations)
    print w
    print datetime.now() - begin
    # Normal equationn method
    print normalEqn(np.hstack((np.ones((X.shape[0],1)),train[:,0:2])),Y)
    
if __name__ == '__main__':
    OneVariable_LinearRegression()
    MultipleVariable_LinearRegression()