# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:37:26 2020

@author: prakh
"""

# IMPORT LIBRARIES 
import numpy as np
import random
import matplotlib.pyplot as plt
from HillClimbing import HC
from SimulatedAnnealing import SA
from LateAcceptedHillClimbing import LAHC
from GeneticAlgorithm import GA,tournament_select,uniform_crossover,real_gaussian_init,real_gaussian_nbr
from RandomSearch import RS
import cma
from pyswarm import pso
import itertools
from BestNeighbours import BestNeighborsAlgo


# IMPORTING DATASET (Location of trees and amount of fruits on them)
filename = 'fruit_picking.txt'
m = 6 # number of bins
data = np.genfromtxt(filename)
T = data[:, 0:2] # x, y locations of trees
weight = data[:, 2] # amount of fruit per tree
n = T.shape[0]

def dist(a, b):
    # Euclidean distance
    return np.sqrt(np.sum((a - b) ** 2))


def facility_location(x):
    x = np.array(x)
    # objective: minimise weighted distance to nearest bin, summed across trees.
    bins = x.reshape((int(len(x)/2), 2))
    total_dist = 0.0
    for t in range(n):
        tree = T[t]
        nearest = min(dist(tree, b) for b in bins)
        total_dist += weight[t] * nearest
    return total_dist


# GENERAL INIT AND NEIGHBOR FUNCTIONS
def real_init(n):
    return np.random.random(n)

def real_nbr(x):
    delta = 0.3
    x = x.copy()
    i = random.randrange(len(x))
    # add a small real constant in range [-delta, delta]
    x[i] = x[i] + (2 * delta * np.random.random() - delta)
    return x

######################################
# UTILITY FUNCTION FOR RANDOM SEARCH #
######################################
def facility_run_RandomSearch(its=10000,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    stop = lambda i, fx: abs(fx) < 0.00001
    x,history = RS(f, lambda: real_init(n), real_nbr, its, stop=stop)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        plt.title("Random seacrh")
        plt.show()
    fbest = facility_location(x)
    return x,fbest

# For Plotting purpose    
# X = facility_run_RandomSearch(its=50000,plot=True)
seedList = [i for i in range(5)]
fvalues=[]
for s in seedList:
    random.seed(s)
    x,fbest = facility_run_RandomSearch(its=50000)
    fvalues.append(fbest)
    
fvalues = np.array(fvalues)   
print("The objective function values for Random search are : ",fvalues)
print("The mean objective function value for Random search is : ",np.mean(fvalues))
print("The standard deviation for objective function for Random search is : ",np.std(fvalues))

################################################
#  UTILITY FUNCTION FOR HILL CLIMBING FUNCTION #
################################################
def facility_run_Hill_Climbing(its=10000,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    stop = lambda i, fx: abs(fx) < 0.00001
    x,history = HC(f, lambda: real_init(n), real_nbr, its, stop=stop)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        plt.title("Hill Climbing")
        plt.show()
    fbest = facility_location(x)
    return x,fbest

"""  
# Single Run for testing    
X = facility_run_Hill_Climbing()
facility_location(X[0])
"""
# For plotting purpose
# X = facility_run_Hill_Climbing(its=2000,plot=True)

seedList = [i for i in range(5)]
fvalues=[]
for s in seedList:
    random.seed(s)
    x,fbest = facility_run_Hill_Climbing(its=50000)
    fvalues.append(fbest)

fvalues = np.array(fvalues)   
print("The objective function values for Hill Climbing are : ",fvalues)
print("The mean objective function value for Hill Climbing is : ",np.mean(fvalues))
print("The standard deviation for objective function for Hill Climbing is : ",np.std(fvalues))

############################################
# UTILITY FUNCTION FOR SIMULATED ANNEALING #
############################################
def facility_run_Simulated_Annealing(T=1,alpha=0.999,maxints=5000,plot=False):
    n=12
    f = lambda x: facility_location(x) 
    bestx, bestfx,history =SA(f,
                      lambda: real_init(n),
                      real_nbr,
                      T,
                      alpha,
                      maxints)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        titleStr = "Simulated Annealing T = "+str(T)+" alpha = "+str(alpha)
        plt.title(titleStr)
        plt.show()
    return bestx,bestfx    

"""
# Single Run for testing 
X = facility_run_Simulated_Annealing()
facility_location(X)
"""
# For plotting purpose
# X = facility_run_Simulated_Annealing(T=1,alpha=0.5,maxints=2000,plot=True)

Tvalues = [1,2]
alphaValues = [0.5,0.9]
for temp,alpha in itertools.product(Tvalues,alphaValues):
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_Simulated_Annealing(T=temp,alpha=alpha,maxints=50000)
        fvalues.append(fbest)

    fvalues = np.array(fvalues)   
    print(f"\n\nThe objective function values for Simulated Annealing (T = {temp} , alpha ={alpha} )  are : ",fvalues)
    print(f"The mean objective function value for Simulated Annealing (T = {temp} , alpha ={alpha} ) is : ",np.mean(fvalues))
    print(f"The standard deviation for objective function for Simulated Annealing (T = {temp} , alpha ={alpha} ) is : ",np.std(fvalues))

####################################################
# UTILITY FUNCTION FOR LATE ACCEPTED HILL CLIMBING #
####################################################
def facility_run_LAHC(L=10,maxiter=10000,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    best, Cbest,history = LAHC(L, maxiter, f, lambda: real_init(n), real_nbr)
    if(plot):
        plt.plot(history[:, 0], history[:, 1])
        plt.xlabel("Iteration"); plt.ylabel("Objective")
        titleStr = "LAHC L = "+str(L)
        plt.title(titleStr)
        plt.show()
    return best,Cbest

"""
# Single Run for testing 
bestx,bestfx = facility_run_LAHC(L=2) 
"""
# For plotting purpose
# X = facility_run_LAHC(L=10,maxiter=3000,plot=True)

Lvalues = [2,10,20]
for L in Lvalues:
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_LAHC(L=L,maxiter=50000)
        fvalues.append(fbest)
        
    fvalues = np.array(fvalues)   
    print(f"\n\nThe objective function values for LAHC (L = {L} )  are : ",fvalues)
    print(f"The mean objective function value for LAHC (L = {L} ) is : ",np.mean(fvalues))
    print(f"The standard deviation for objective function for LAHC (L = {L} ) is : ",np.std(fvalues))


##########################################
# UTILITY FUNCTION FOR GENETIC ALGORITHM #
##########################################
def facility_run_Genetic_Algorithm(popsize = 100,ngens = 100,pmut = 0.1,tsize = 2,plot=False):
    n = 12
    f = lambda x: facility_location(x) 
    bestf, best, h = GA(f,
                    lambda: real_gaussian_init(n),
                    real_gaussian_nbr,
                    uniform_crossover,
                    lambda pop, popfit: tournament_select(pop, popfit, tsize),
                    popsize,
                    ngens,
                    pmut
                    )
    if(plot):
        plt.plot(h[:, 1], h[:, 2])
        plt.xlabel("Iterations");plt.ylabel("Fitness")
        titleStr = "Genetic Algo P = "+str(popsize)+" G = "+str(ngens)+" M = "+str(pmut)+" T = "+str(tsize)
        plt.title(titleStr)
        plt.show()
        plt.close()
    
        # plot std fit against number of individuals 
        plt.plot(h[:, 1], h[:, -1])
        plt.xlabel("Iterations");plt.ylabel("Standard deviation (fitness)")
        plt.title(titleStr)
        plt.show()
    return best,bestf

"""
# Single Run for testing 
X = facility_run_Genetic_Algorithm(popsize = 100,ngens = 100,pmut = 0.1,tsize = 2)
facility_location(X[0])
"""
# For plotting purpose
# X = facility_run_Genetic_Algorithm(popsize = 200,ngens = 250,pmut = 0.2,tsize = 2,plot=True)

populationSize = [100,200]
mutationRatio = [0.1,0.2]
tournamentSize = [2]

for pop,pmut,tsize in itertools.product(populationSize,mutationRatio,tournamentSize):
    seedList = [i for i in range(5)]
    fvalues=[]
    # considering a fitness evaluation budget of 50000 budget = populationSize*noOfGeneration
    gen = int(50000/pop)
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_Genetic_Algorithm(popsize = pop,ngens = gen,pmut = pmut,tsize = tsize)
        fvalues.append(fbest)
    fvalues = np.array(fvalues)   
    print(f"\n\nThe objective function values for Genetic Algo ( Pop Size : {pop} ,  Gen : {gen} , Mutation ratio :{pmut} , Tournament size : {tsize} are : ",fvalues)
    print(f"The mean objective function value for Genetic Algo ( Pop Size : {pop} ,  Gen : {gen} , Mutation ratio :{pmut} , Tournament size : {tsize} is : ",np.mean(fvalues))
    print(f"The standard deviation for objective function for Genetic Algo ( Pop Size : {pop} ,  Gen : {gen} , Mutation ratio :{pmut} , Tournament size : {tsize} is : ",np.std(fvalues))


#####################################################
# UTILITY FUNCTION FOR COVARIANCE MATRIX ADAPTATION #
#####################################################
def estimate_full(pop):
    # estimate the distribution of the samples in `pop`: the model
    # consists of the mean (a vector) and full covariance matrix.
    mu = np.mean(pop, axis=0)
    sigma = np.cov(pop, rowvar=False)
    return mu, sigma

def facility_run_CMA(popsize=100):
    # Reference : http://cma.gforge.inria.fr/html-pythoncma/frames.html
    # The CMA Evolution Strategy: A Tutorial : https://arxiv.org/pdf/1604.00772.pdf (From: Nikolaus Hansen)
    n = 12
    f = lambda x: facility_location(x) 
    pop = [real_init(n) for i in range(popsize)]
    mu,sigma = estimate_full(pop)
    es=cma.CMAEvolutionStrategy(mu,1,
                                {'bounds': [-np.inf, np.inf],
                                 'seed':234,
                                 'popsize':popsize,
                                 'maxiter':1000})
    es.optimize(f)    
    xbest, fbest, evals_best, evaluations, iterations, xfavorite, stds, stop = es.result
    return xbest,fbest

"""
# Single Run for testing 
X = facility_run_CMA()
facility_location(X[0])
"""

populationSize = [10,100,200]
for pop in populationSize:
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_CMA(popsize=pop)
        fvalues.append(fbest)
        
    fvalues = np.array(fvalues)   
    print(f"\n\nThe objective function values for CMA ( Pop Size : {pop} )  are : ",fvalues)
    print(f"The mean objective function value for CMA ( Pop Size : {pop} )  is : ",np.mean(fvalues))
    print(f"The standard deviation for objective function for CMA ( Pop Size : {pop} )  is : ",np.std(fvalues))

#################################################
# UTILITY FUNCTION FOR PARTICLE SWARM ALGORITHM #
#################################################
# Tried implementing PSO from scratch using below link but was not giving good results (worse then random search) :(
# PSO from scratch : https://medium.com/analytics-vidhya/implementing-particle-swarm-optimization-pso-algorithm-in-python-9efc2eb179a6
# DEFINE COST FUNCTIONS

def facility_run_PSO(maxiter=1000,omega=0.5,phip =0.5,phig =0.5):
    # Reference : https://pythonhosted.org/pyswarm/ (PySwarm Library for Python's PMO Implementation)
    
    # maxiter : The maximum number of iterations for the swarm to search (Default: 100)
    # omega : Particle velocity scaling factor (Default: 0.5)
    # phip  : Scaling factor to search away from the particle’s best known position (Default: 0.5)
    # phig : Scaling factor to search away from the swarm’s best known position (Default: 0.5)
    n = 12
    lb = [0 for i in range(n)] # To be in range of farm (lower bound)
    ub = [8 for i in range(n)] # To be in range of farm (upper bound)
    xoft,fopt=pso(facility_location,lb,ub,maxiter=maxiter)
    #xoft,fopt= pso(facility_location,lb,ub,omega=omega,phip=phip,phig=phig,maxiter=maxiter)
    return xoft,fopt

""" 
# Single Run for testing 
X,f=facility_run_PSO()
"""
# Basic formula's for PSO
# V = W*V + cp*rand()(pbest-x) + cg*rand()(gbest-x)
# X = X +V
W = [0.5,1]
cp = [0.5,0.9] # not exactly cp of formula 
cg = [0.5,0.9] # not exactly cg of formula

for w,p,g in itertools.product(W,cp,cg):
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        x,fbest = facility_run_PSO(omega=w,phip=cp,phig=cg)
        fvalues.append(fbest)
    fvalues = np.array(fvalues)   
    print(f"\n\nThe objective function values for PSO ( w = {w} , cp ={p} , cg ={g} ) are : ",fvalues)
    print(f"The mean objective function value for PSO ( w = {w} , cp ={p} , cg ={g} ) is : ",np.mean(fvalues))
    print(f"The standard deviation for objective function for PSO ( w = {w} , cp ={p} , cg ={g} ) is : ",np.std(fvalues))


#################################################################################
# UTILITY FUNCTION FOR BEST NEIGHBOUR ALGORITHM  - TRIED CREATING NEW ALGORITHM #
#################################################################################
# Algorithm is implemented in  BestNeighbours.py   
def facility_run_Best_Neighbors_Algo(its=50,c_hyperparam=3):
    n = 12
    f = lambda x: facility_location(x) 
    fbest = BestNeighborsAlgo(f, lambda: real_init(n), its,c=c_hyperparam) # Set hyperparameter c as 3 (was giving good results)
    return fbest


for c_h in [3,4,5,6]:
    seedList = [i for i in range(5)]
    fvalues=[]
    for s in seedList:
        random.seed(s)
        fbest =  facility_run_Best_Neighbors_Algo(its=50,c_hyperparam=c_h)
        fvalues.append(fbest)
    
    fvalues = np.array(fvalues)   
    print(f"\n\nThe objective function values for Best Neighbours algo  ( c = {c_h} )  are : ",fvalues)
    print(f"The mean objective function value for Best Neighbours algo  ( c = {c_h} ) is : ",np.mean(fvalues))
    print(f"The standard deviation for objective function for Best Neighbours algo  ( c = {c_h} ) is : ",np.std(fvalues))
