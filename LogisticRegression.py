import numpy as np
import pandas as pd
from datetime import datetime

'''
The dataset comes from the following links:
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw3%2Fhw3_train.dat
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw3%2Fhw3_test.dat
'''

def sigmod(num):
    return 1.0/(1.0+np.exp(-num))

def GradientDescent(X,Y,w,eta,iterations):
    '''
    The gradient is:
    (1/n)*\sum_{n=1}^N[sigmod(-y_nX_n*W^T)*(-y_n*X_n)]
    '''
    for t in xrange(iterations):
        ans = sigmod(-Y*X.dot(w))
        grad = [ans[i,:]*(-Y[i,:]*X[i,:]) for i in xrange(X.shape[0])]
        w = w - (eta*sum(grad)/X.shape[0]).reshape(w.shape[0],1)
    return w

def StochasticGradientDescent(X,Y,w,eta,iterations):
    for t in xrange(iterations):
        index = np.mod(t,X.shape[0])
        ans = sigmod(-Y[index,:]*X[index,:].dot(w))
        w = w - (eta*ans*(-Y[index,:]*X[index,:])).reshape(w.shape[0],1)
    return w

def LogisticRegression():
    train = pd.read_csv("D://ntumlone-hw3-hw3_train.dat",header = None, sep = ' ')
    train = np.array(train)  
    X = train[:,1:21]
    Y = train[:,21:]
    
    test = pd.read_csv("D://ntumlone-hw3-hw3_test.dat",header = None, sep = ' ')
    test = np.array(test)      
    test_x = test[:,1:21]
    test_y = test[:,21:]    
    
    # NOTING: change eta you will get different results
    eta = 0.001
    iterations = 2000
    w = np.zeros((20,1))
    
    begin = datetime.now()
    w1 = GradientDescent(X,Y,w,eta,iterations)
    print datetime.now() - begin
    
    # predict on the test dataset with the weight get by GD
    test_ans = sigmod(test_x.dot(w1))
    base = np.ones((test_x.shape[0],1))/2.0
    result = sum(np.sign(test_ans - base) != test_y)
    # print the 0-1 error with GD
    print result/float(test_x.shape[0])
    
    begin = datetime.now()
    w2 = StochasticGradientDescent(X,Y,w,eta,iterations)
    print datetime.now() - begin
    # predict on the test dataset with the weight get by SGD
    test_ans = sigmod(test_x.dot(w2))
    base = np.ones((test_x.shape[0],1))/2.0
    result = sum(np.sign(test_ans - base) != test_y)
    # print the 0-1 error with SGD
    print result/float(test_x.shape[0])
    
if __name__ == '__main__':
    LogisticRegression()
    
# The results shows that GD and SGD will get the same result, but the time cost gap is huge.