import numpy as np
import pandas as pd
from datetime import datetime

'''
This code mainly shows how to build a CV model for tuning parms

The datasets come from the following link:
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw4%2Fhw4_train.dat
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw4%2Fhw4_test.dat
'''

def RidgeRegression(lamb = 10):
    # change the lamd will get different results
    train = pd.read_csv("D://ntumlone-hw4-hw4_train.dat",header = None, sep = ' ')
    train = np.array(train)  
    X = train[:,0:2]
    Y = train[:,2:]
    # add x_0
    X = np.hstack((np.ones((X.shape[0],1)),X))
    
    test = pd.read_csv("D://ntumlone-hw4-hw4_test.dat",header = None, sep = ' ')
    test = np.array(test)      
    test_x = test[:,0:2]
    test_y = test[:,2:]
    # add x_0
    test_x = np.hstack((np.ones((test_x.shape[0],1)),test_x))
    
    # Equation solution : w_reg = (z^T*Z + lambda*I)^-1Z^Ty
    w = np.linalg.inv(X.T.dot(X) + lamb*np.eye(3)).dot(X.T.dot(Y))
    train_error = sum(np.sign(X.dot(w)) != Y)/float(X.shape[0])
    test_error = sum(np.sign(test_x.dot(w)) != test_y)/float(test_x.shape[0])
    
    print "training dataset error",train_error
    print "testing dataset error",test_error
    
def RigdeRegressionWithCV():
    # use cross validation to choose lamdba
    # split the training set into 5 folds
    train = pd.read_csv("D://ntumlone-hw4-hw4_train.dat",header = None, sep = ' ')
    train = np.array(train)  
    X = train[:,0:2]
    Y = train[:,2:]
    # add x_0
    X = np.hstack((np.ones((X.shape[0],1)),X))

    lambs = [100,10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001]
    errors = []
    for lamb in lambs:
        #print "-----"
        #print lamb
        cv_error = 0
        for i in xrange(5):
            # print i
            # construct validation data
            tempx = X[i*40:(i+1)*40,:]
            tempy = Y[i*40:(i+1)*40,:]
            # construct the new train data
            restx_1 = X[(i+1)*40:,:]
            restx_2 = X[:i*40,:]
            resty_1 = Y[(i+1)*40:,:]
            resty_2 = Y[:i*40,:]
            restx = np.vstack((restx_1,restx_2))
            resty = np.vstack((resty_1,resty_2))
            # calculate the w
            w = np.linalg.inv(restx.T.dot(restx) + lamb*np.eye(3)).dot(restx.T.dot(resty))
            cv_error = cv_error + sum(np.sign(tempx.dot(w)) != tempy)/float(tempx.shape[0])
        #print cv_error/5
        errors.append(cv_error/5)
    return lambs[errors.index(min(errors))] 
    # the cv_error shows the best lambda is 1e-08
    # then you can use this lamb to build the regression model
    
if __name__ == "__main__":
    print "-------before CV"
    RidgeRegression()
    lamb = RigdeRegressionWithCV()
    print "-------use CV"    
    RidgeRegression(lamb)