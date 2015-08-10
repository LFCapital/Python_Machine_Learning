import numpy as np
import pandas as pd
from datetime import datetime
'''
IMPORTANT:
the key update rule of PLA is:
w_{t+1} = w_{t} + y_n(t)*X_n(t)
only the wrong points will be updated

This file implement two kinds of PLA, namely naive cycles PLA, fixed random cycles PLA

The training dataset come from the following link:
https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat
'''

#read data and do the initialization
train = pd.read_csv("D:\\ntumlone_hw1_hw1_15_train.dat",header = None, sep = ' ');
train = np.array(train)
train_x = train[:,0:4]
train_y = train[:,4:]

# get the row number
train_num = train_x.shape[0]
# add constant 1 to each row
train_x = np.hstack((np.ones((train_num,1)),train_x))

def naive_PLA(train_x,train_y):
    # weight vector
    w = np.zeros((5,1))
    # update counter
    wcount = 0
    # exit flag
    flag = 1
    while flag:
        # if count remains zero after the for loop, it means all points are correctly classified
        count = 0
        for t in xrange(train_num):
            # set sign(0) = -1        
            p_value = 1 if np.sign(train_x[t,:].dot(w))>0 else -1
            if p_value != train_y[t,:]:
                # record updtae times
                wcount = wcount + 1
                # update the w
                w = w + (train_y[t,:]*train_x[t:t+1,:]).T
                count = count + 1
        # print the update count in each turn
        print count
        # if no mistakes, then break
        if count == 0:
            break
    # print the total update numbers  
    print "The total update times for navie PLA is",wcount

def random_PLA(train_x,train_y):
    wcount = 0
    times = 2000
    for time_index in xrange(times):
        w = np.zeros((5,1))
        flag = 1
        while flag:
            count = 0
            index = np.random.permutation(train_x.shape[0])
            for t in index:
                p_value = 1 if np.sign(train_x[t,:].dot(w))>0 else -1
                if p_value != train_y[t,:]:
                    # record updtae times
                    wcount = wcount + 1
                    # update the w
                    w = w + (train_y[t,:]*train_x[t:t+1,:]).T
                    count = count + 1
            # if no mistakes, then break
            if count == 0:
                break
    # print the total update numbers  
    print "The total update times for random PLA is",wcount/float(times)

naive_PLA(train_x,train_y)

begin = datetime.now()
random_PLA(train_x,train_y)
print "The random PLA cost",datetime.now()-begin

'''
1.Speed: Naive cycles PLA VS Random cycles PLA
The results shows that random cycles can achieve better performance

2.Speed: Python VS Matlab
I dit some experiments on Matlab, the Matlab speed performance on the same scale problem is much faster than python
'''