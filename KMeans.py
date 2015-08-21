import numpy as np
import pandas as pd
'''
This codes shows the principle of K-means algorithm.

The dataset comes from the following link:
https://d396qusza40orc.cloudfront.net/ntumltwo/hw4_data/hw4_kmeans_train.dat
'''

def K_Means(k = 2):
    train = pd.read_csv("D://hw4_kmeans_train.dat",header = None, sep = ' ')
    train = np.array(train)[:,0:9]
    
    times = 500
    error = 0
    
    for t in xrange(times):
        print t
        # initialize the k centers
        result_center = np.zeros((k,train.shape[1]))
        # random pick the initial centers
        index = np.random.permutation(train.shape[0])
        # or index = np.random.randint(0,train.shape[0],k)
        for j in xrange(k):
            result_center[j,:] = train[index[j],:]
            
        # record each point belong to which cluster
        result_belong = np.zeros((k,train.shape[0]))
        
        while True:
            # record each point belong to which cluster in every round
            temp_belong = np.zeros((k,train.shape[0]))
            # calculate the each point belong to which centers
            for i in xrange(train.shape[0]):
                # get the i's point to all centers distance
                distance = [sum((train[i,:]-result_center[ki,:])**2) for ki in xrange(k)]
                # find the minuium
                min_d = min(distance)
                # get the index
                ind = distance.index(min_d)
                # record i's point to the index center
                temp_belong[ind,i] = 1
            # initial temp_centers
            temp_center = np.zeros((k,train.shape[1]))
            # recalculate the k centers
            for ki in xrange(k):
                t_result = sum([temp_belong[ki,i]*train[i,:] for i in xrange(train.shape[0])])/float(sum(temp_belong[ki,:]))
                temp_center[ki,:] = t_result
            # judge whether the converge condition
            if sum(sum(temp_center == result_center)) == k*train.shape[1]:
                result_belong = temp_belong
                break
            else:
                # update the k center
                result_center = temp_center
        
        #--------calculte the error
        dis = 0
        for ki in xrange(k):
            for i in xrange(train.shape[0]):
                if result_belong[ki,i] == 1:
                    dis = dis + sum((train[i,:]-result_center[ki,:])**2)
        error = error + dis/float(train.shape[0])
        print error/t

if __name__ == "__main__":
    K_Means()
    K_Means(10)