import numpy as np
import pandas as pd
from datetime import datetime
'''
IMPORTANT:
This file implement the Pocket algorithm.
Pocket is nearly the same as PLA, the key idea of poket is: keep the best performance w.

The dataset come from the following link:
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_train.dat
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_18_test.dat
'''

#read data and do the initialization
train = pd.read_csv("D:\\ntumlone_hw1_hw1_18_train.dat",header = None, sep = ' ');
train = np.array(train)
train_x = train[:,0:4]
train_y = train[:,4:]

test = pd.read_csv("D:\\ntumlone_hw1_hw1_18_test.dat",header = None, sep = ' ');
test = np.array(test)
test_x = test[:,0:4]
test_y = test[:,4:]

# get the row number
train_num = train_x.shape[0]
test_num = test_x.shape[0]

# add constant 1 to each row
train_x = np.hstack((np.ones((train_num,1)),train_x))
test_x = np.hstack((np.ones((test_num,1)),test_x))

def Pocket(train_x,train_y,test_x,test_y):
    times = 2000
    each_round_times = 50
    error = 0
    for time_index in xrange(times):
        # print time_index
        # weight vector
        w = np.zeros((5,1))
        # best weight vector
        wbest = np.zeros((5,1))
        # mistake counter
        count = 0
        # random update cycle index
        index = np.random.permutation(train_x.shape[0])
        flag = 1
        while flag:
            for t in index:
                p_value = 1 if np.sign(train_x[t,:].dot(w))>0 else -1
                if p_value != train_y[t,:]:
                    # update the w
                    w = w + (train_y[t,:]*train_x[t:t+1,:]).T
                    count = count + 1
                    # update the best
                    if sum(np.sign(train_x.dot(w)) == train_y)>sum(np.sign(train_x.dot(wbest)) == train_y):
                        wbest = w
                    if count == each_round_times:
                        flag = 0
                        break
        # record the error rate for testset in each turn
        error = error + int(sum(np.sign(test_x.dot(wbest)) != test_y))/float(test_x.shape[0])
    print "The error rate for Pocket is", error/float(times)

begin = datetime.now()
Pocket(train_x,train_y,test_x,test_y)
print "The Pocket algoritm cost",datetime.now()-begin