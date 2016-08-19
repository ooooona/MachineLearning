import numpy as np
import math as math
import copy as copy
import random as random
import time as tm
import matplotlib.pyplot as plt

def calGuassianProb(x, Mu, Sigma):
	exponent = math.exp( -(x-Mu)**2 / float(2 * (Sigma**2)) )
	return exponent / float(Sigma * math.sqrt(2 * math.pi))
 
def genGMMData(m, GausParam, ratio):
    k = int(GausParam.shape[0])        #n_components
    X = np.zeros((m,))                 #samples
    z = np.random.multinomial(m, ratio)#n_components_sampling
    xCount = 0
    for i in xrange(k):
        num = z[i]
        X[xCount:xCount + num] = np.random.normal(GausParam[i,0], GausParam[i,1], num)
        xCount += num
    return X

def E_Step(X, GausParam, Phi):
    m = int(X.shape[0])           #n_samples
    k = int(GausParam.shape[0])   #n_components
    W = np.zeros((m, k))          
    for i in xrange(m):
        demon = 0.
        for j in xrange(k):
            demon += float(Phi[j]) * calGuassianProb(X[i], GausParam[j,0], GausParam[j,1])
        for j in xrange(k):
            mole = float(Phi[j]) * calGuassianProb(X[i], GausParam[j,0], GausParam[j,1])
            W[i,j] = mole/demon
            #print j, float(Phi[j]) , calGuassianProb(X[i], GausParam[j,0], GausParam[j,1])
    return W
            
def M_Step(X, W):
    m = int(X.shape[0]) #n_samples
    k = int(W.shape[1]) #n_components
    new_GausParam = np.zeros((k, 2))
    new_Phi = np.zeros((k,))
    for j in xrange(k):
        Nj = np.sum(W[:,j])
        new_Phi[j] = Nj/float(m)
        new_GausParam[j, 0] = X.dot(W[:,j])/float(Nj)
        new_GausParam[j, 1] = ((X-new_GausParam[j,0])**2).dot(W[:,j])/float(Nj)
    return new_GausParam, new_Phi
    
def EM_Run():
    k = 3        #n_components
    m = 100      #n_samples
    ratio = [0.5, 0.25, 0.25]    #prob_of_component
    GausParam = np.zeros((k,2))  
    GausParam[0, 0] = 10.0
    GausParam[1, 0] = 100.0
    GausParam[2, 0] = 50.0
    GausParam[0, 1] = GausParam[1, 1] = GausParam[2, 1] = 3.0
    X = genGMMData(m, GausParam, ratio)    #samples
    iter_num = 10000
    Epsilon = 0.00001
    old_GausParam = np.zeros((k,2))       #initial_gaussian
    old_GausParam[:,0] = (150*np.random.random(k))
    old_GausParam[:,1] = (10*np.random.random(k))
    old_Phi = np.random.random(k)         #initial_prob_od_component
    old_Phi = old_Phi/sum(old_Phi)
#    old_GausParam = copy.deepcopy(GausParam)
#    old_Phi = np.array(ratio)
    print 'GausParam = ' , old_GausParam
    print 'Phi = ', old_Phi
    print X
    tm.sleep(1)
    for i in xrange(iter_num):
        W = E_Step(X, old_GausParam, old_Phi)
        new_GausParam, new_Phi = M_Step(X, W)
        print i, 'guassian param = ', new_GausParam
        tm.sleep(1)
        if(sum(abs(new_GausParam[:,0] - old_GausParam[:,0])) < Epsilon):
            print i , 'break', new_GausParam, old_GausParam
            break
        old_GausParam = copy.deepcopy(new_GausParam)
        old_Phi = copy.deepcopy(new_Phi)
    print GausParam, ratio
    print old_GausParam, old_Phi
    
    xias = np.arange(0.0, 120.0, 1.0)  
    for i in xrange(k):
        Mu = old_GausParam[i][0]
        Sigma = old_GausParam[i][1]
        print Mu, Sigma
        yias = 100*(np.exp(-(xias - Mu)**2 / (2 * Sigma**2)) / (Sigma * np.sqrt(2 * np.pi)))
        plt.plot(xias, yias, color='r', linewidth=2)
        plt.show()
        
def EM_Run1():
    k = 3
    m = 1000
    ratio = [1.0/2, 1.0/4, 1.0/4]
    GausParam = np.zeros((k,2))
    GausParam[0, 0] = 10.0
    GausParam[1, 0] = 40.0
    GausParam[2, 0] = 25.0
    GausParam[0, 1] = GausParam[1, 1] = GausParam[2, 1] = 6.0
    X = genGMMData(m, GausParam, ratio)
    iter_num = 10000
    Epsilon = 0.00001
    W = np.random.multinomial(1, [0.3,0.3,0.4], m)
    print 'W = ', W
    tm.sleep(1)
    old_GausParam = None
    old_Phi = None
    for i in xrange(iter_num):
        if i != 0:
            W = E_Step(X, old_GausParam, old_Phi)
        new_GausParam, new_Phi = M_Step(X, W)
        print i, 'guassian param = ', new_GausParam
        if(i!=0 and sum(abs(new_GausParam[:,0] - old_GausParam[:,0])) < Epsilon):
            print i , 'break', new_GausParam, old_GausParam
            break
        old_GausParam = copy.deepcopy(new_GausParam)
        old_Phi = copy.deepcopy(new_Phi)
    print GausParam, ratio
    print old_GausParam, old_Phi
    plt.hist(X,50)
    plt.show()
    

EM_Run()
        
            