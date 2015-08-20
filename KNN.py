import numpy as np
import pandas as pd
'''
This code shows how to build a KNN classification model.
The key idea of KNN is the majority rule, given a point and search his K neighbors
find out the majority class labels of his neighbors.

The optimization data structure for KNN is KD-TREE.

The dataset for this code comes from the following links:
https://d396qusza40orc.cloudfront.net/ntumltwo/hw4_data/hw4_knn_train.dat
https://d396qusza40orc.cloudfront.net/ntumltwo/hw4_data/hw4_knn_test.dat

'''
def KNN(K = 1):
    train = pd.read_csv("D://hw4_knn_train.dat",header = None, sep = ' ')
    train = np.array(train)
    X = train[:,0:9]
    Y = train[:,9:]

    test = pd.read_csv("D://hw4_knn_test.dat",header = None, sep = ' ')
    test = np.array(test)
    testX = test[:,0:9]
    testY = test[:,9:]

    error = 0
    for i in xrange(testX.shape[0]):
        # calculate the distance        
        distance = [sum((X[j,:] - testX[i,:])**2) for j in xrange(X.shape[0])]
        # initialize the label value    
        label = [Y[j,:] for j in xrange(X.shape[0])]
        # construct the label distance tuple        
        mix = zip(label,distance)
        # sort the distance
        mix.sort(key = lambda x:x[1])
        #classify the test point
        result = np.sign(sum([v[0] for v in mix[0:K]]))
        # calculate the error        
        if testY[i,:] != result:
            error = error + 1
    print "The test error is",error/float(testX.shape[0])

if __name__ == "__main__":
    KNN()
    KNN(5)
    KNN(10)
'''
 You can change the K to see the different results.
'''