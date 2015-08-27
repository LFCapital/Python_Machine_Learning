import numpy as np
import pandas as pd
from DecisionTree import *
'''
This codes shows the preliminary randomforest algorithm.

The dataset is the same as the DecisionTree.py, namely:

https://d396qusza40orc.cloudfront.net/ntumltwo/hw3_data/hw3_train.dat
https://d396qusza40orc.cloudfront.net/ntumltwo/hw3_data/hw3_test.dat
'''
train = pd.read_csv("D://hw3_train.dat",header = None, sep = ' ')
train = np.array(train)
train_x = train[:,0:2]
train_y = train[:,2:]

test = pd.read_csv("D://hw3_test.dat",header = None, sep = ' ')
test = np.array(test)
test_x = test[:,0:2]
test_y = test[:,2:]

times = 1
ntree = 10

train_result_tree = []
train_result_rf = np.zeros((times,1))
test_result_rf = np.zeros((times,1))

for time in xrange(times):
    N = train_x.shape[0]
    Num_feature = train_x.shape[1]
    
    newdatax = np.zeros((N,Num_feature))
    newdatay = np.zeros((N,1))
    pre_result_rf_train = np.zeros((N,ntree))
    pre_result_rf_test = np.zeros((test_x.shape[0],ntree))
    
    # start randomforest
    for tree in xrange(ntree):
        print tree
        # bootstap operations
        bootstrap = np.random.randint(N, size=N)
        for i in xrange(N):
            newdatax[i,:] = train_x[bootstrap[i],:]
            newdatay[i,:] = train_y[bootstrap[i],:]
        # building a decision tree
        DNode = DTtree(newdatax,newdatay,0)
        # predict for training dataset
        pre_result =  np.array([Predict(DNode,train_x[i,:]) for i in xrange(train_x.shape[0])])
        pre_result_rf_train[:,tree] = pre_result        
        
        # record this tree's error in training dataset
        train_result_tree.append(sum(pre_result.reshape(train_x.shape[0],1)!=train_y)/float(train_y.shape[0]))

        # predict for testing dataset
        pre_result =  np.array([Predict(DNode,test_x[i,:]) for i in xrange(test_x.shape[0])])
        pre_result_rf_test[:,tree] = pre_result
    rfpred_train = np.sign(np.sum(pre_result_rf_train, axis=1))
    rfpred_test = np.sign(np.sum(pre_result_rf_test, axis=1))
    train_result_rf[time] =  sum(rfpred_train.reshape(rfpred_train.shape[0],1)!=train_y)/float(train_y.shape[0])
    test_result_rf[time] = sum(rfpred_test.reshape(rfpred_test.shape[0],1)!=test_y)/float(test_y.shape[0])

print "AVG Tree Train Error is",np.array(train_result_tree).mean()
print "AVG RF Train Error is",train_result_rf.mean()
print "AVG RF Test Error is",test_result_rf.mean()

'''
Remark: This code just shows the princple of randomforest, the speed of the implemention is very slow.
Some tools can help you to speed up the performance.
'''
