# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:18:05 2020

@author: prakh
"""

# IMPORT LIBRARIES 
import random
import collections

# Creates n neighbours for n-dimensional input vector (distance is random from original input vector)
def nearby(X,c):
    newList=list()
    X_copy = X.copy()
    for i in range(len(X)):
        X_copy[i] = X[i] + (c*random.random()) # c is hyperparameter 
        newList.append(X_copy)
        X_copy = X.copy()
    return newList

# nearby([1,2,3,4,5],1)
  
# Note : the code currently is poorly written, will try making it better this weekend
def BestNeighborsAlgo(f,init,its,c=3,m=100):
    x = init() # start with a initial X
    domainList = list() # stores our population (we will limit out poulation to m, by default m = 100 so our population never goes beyond m)
    domainList.append(x)
    for i in range(its):
        newDomainList = list()
        for e in domainList:
            nearList = nearby(e,c) # for every x in domainList generate its n-dimensional neighbours
            newDomainList.extend(nearList) 
            newDomainList.append(e) # append the original x too 
            
        D = dict()
        # create a dict of objective_value : x
        for d in newDomainList:
            D[f(d)]=d
          
        # sort dictionary based on keys(objective function)
        P = collections.OrderedDict(sorted(D.items()))
        size = min(len(P),m)
        # make next population as empty
        domainList = list()
        # our next domain list (population) will have only best m vectors (in mimimization case lowest objective value's one)
        ip=0
        for p in P.values():
            ip=ip+1
            if ip>size:
                break
            domainList.append(p)
        # this domainList will act as next population   
    Y = [f(y) for y in domainList]
    return min(Y)
