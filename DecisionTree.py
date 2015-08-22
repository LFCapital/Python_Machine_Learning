import numpy as np
import pandas as pd
'''
This code shows how to build a C&RT tree.

The dataset for this code comes from the following links:
https://d396qusza40orc.cloudfront.net/ntumltwo/hw3_data/hw3_train.dat
https://d396qusza40orc.cloudfront.net/ntumltwo/hw3_data/hw3_test.dat
'''

class Node(object):
    def __init__(self):
        self.leaf = None
        self.value = 0
        self.feature = None
        self.theta = None
        self.lefttree = None
        self.righttree = None
        
    def setNode(self,flag):
        self.leaf = flag

    def setValue(self,value):
        self.value = value
        
    def setParams(self,feature,theta):
        self.feature = feature
        self.theta = theta
        
    def setLeftTree(self,left):
        self.lefttree = left
        
    def setRightTree(self,right):
        self.righttree = right
      
def DTtree(train_x,train_y,depth):
    node = Node()
    n_node = train_y.shape[0]
    if sum(train_y) == n_node or sum(train_y) == -n_node:
        node.setNode(True)
        if sum(train_y) == n_node:
            node.setValue(1)
        else:
            node.setValue(-1)
    else:
        node.setNode(False)
        low_result = np.inf
        theta_result = 0
        feature_result = -1
        for f in xrange(train_x.shape[1]):
            temp = train_x[:,f].copy()
            temp.sort()
            temp2 = temp[1:,]
            theta = (temp[0:(n_node-1)]+temp2)/2      
            
            for t in xrange(theta.shape[0]):
                ans = np.sign(train_x[:,f] - theta[t])
                p_index = (ans == 1)
                n_index = (ans == -1)
                t_y = pd.DataFrame(train_y)
                p_y = t_y[p_index]
                n_y = t_y[n_index]

                p_total = float(p_y.size)
                n_total = float(n_y.size)
                p_imputy = 1 - (sum(np.array(p_y == 1))/p_total)**2 - (sum(np.array(p_y == -1))/p_total)**2
                n_imputy = 1 - (sum(np.array(n_y == 1))/n_total)**2 - (sum(np.array(n_y == -1))/n_total)**2

                imputy = p_total*p_imputy + n_total*n_imputy

                if imputy < low_result:
                    low_result = imputy
                    theta_result = theta[t]
                    feature_result = f

        #print feature_result,theta_result,depth
        node.setParams(feature_result,theta_result)
        line = np.sign(train_x[:,feature_result] - theta_result)
        
        left_index = (line == 1)
        right_index = (line == -1)

        temp_x = pd.DataFrame(train_x)
        temp_y = pd.DataFrame(train_y)
        
        left_x = np.array(temp_x[left_index])
        left_y = np.array(temp_y[left_index])
        right_x = np.array(temp_x[right_index])
        right_y = np.array(temp_y[right_index])

        node.setLeftTree(DTtree(left_x,left_y,depth+1))
        node.setRightTree(DTtree(right_x,right_y,depth+1))
    return node

def NodeCount(Nodex):
    if Nodex.leaf:
        count  = 0
    else:
        count = 1 + NodeCount(Nodex.lefttree) + NodeCount(Nodex.righttree)
    return count

def Predict(Nodex,data):
    if Nodex.leaf:
        predict = Nodex.value
    else:
        feature = Nodex.feature
        theta = Nodex.theta
        line = np.sign(data[feature] - theta)
        if line > 0 :
            predict = Predict(Nodex.lefttree,data)
        else:
            predict = Predict(Nodex.righttree,data)
    return predict

def CART():
    train = pd.read_csv("D://hw3_train.dat",header = None, sep = ' ')
    train = np.array(train)
    train_x = train[:,0:2]
    train_y = train[:,2:]
    test = pd.read_csv("D://hw3_test.dat",header = None, sep = ' ')
    test = np.array(test)
    test_x = test[:,0:2]
    test_y = test[:,2:]
    # building the decision tree
    DNode = DTtree(train_x,train_y,0)
    #print NodeCount(DNode)
    
    # calculate the test dataset error
    pre_result =  np.array([Predict(DNode,test_x[i,:]) for i in xrange(test_x.shape[0])])
    pre_result =  pre_result.reshape(1000,1)  
    error = sum(pre_result!=test_y)/float(test_x.shape[0])
    print "Test dataset error is",error
    
if __name__ == "__main__":
    CART()