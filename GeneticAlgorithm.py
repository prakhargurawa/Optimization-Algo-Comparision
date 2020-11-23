# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:32:40 2020

@author: prakh
"""
# IMPORT LIBRARIES
import numpy as np
import random

def real_gaussian_init(n):
    return np.random.normal(size=n)

def real_gaussian_nbr(x):
    delta = 0.1
    x = x.copy()
    # draw from a Gaussian
    x = x + delta * np.random.randn(len(x))
    return x

def stats(gen, popfit):
    # let's return the generation number and the number
    # of individuals which have been evaluated
    return gen, (gen+1) * len(popfit), np.min(popfit), np.mean(popfit), np.median(popfit), np.max(popfit), np.std(popfit)
   

# this GA is for minimisation
def GA(f, init, nbr, crossover, select, popsize, ngens, pmut):
    history = []
    # make initial population, evaluate fitness, print stats
    pop = [init() for _ in range(popsize)]
    popfit = [f(x) for x in pop]
    history.append(stats(0, popfit))
    for gen in range(1, ngens):
        # make an empty new population
        newpop = []
        # elitism
        bestidx = min(range(popsize), key=lambda i: popfit[i])
        best = pop[bestidx]
        newpop.append(best)
        while len(newpop) < popsize:
            # select and crossover
            p1 = select(pop, popfit)
            p2 = select(pop, popfit)
            c1, c2 = crossover(p1, p2)
            # apply mutation to only a fraction of individuals
            if random.random() < pmut:
                c1 = nbr(c1)
            if random.random() < pmut:
                c2 = nbr(c2)
            # add the new individuals to the population
            newpop.append(c1)
            # ensure we don't make newpop of size (popsize+1) - 
            # elitism could cause this since it copies 1
            if len(newpop) < popsize:
                newpop.append(c2)
        # overwrite old population with new, evaluate, do stats
        pop = newpop
        popfit = [f(x) for x in pop]
        history.append(stats(gen, popfit))
    bestidx = np.argmin(popfit)
    return popfit[bestidx], pop[bestidx],np.array(history)

 
def tournament_select(pop, popfit, size):
    # To avoid re-calculating f for same individual multiple times, we
    # put fitness evaluation in the main loop and store the result in
    # popfit. We pass that in here.  Now the candidates are just
    # indices, representing individuals in the population.
    candidates = random.sample(list(range(len(pop))), size)
    # The winner is the index of the individual with min fitness.
    winner = min(candidates, key=lambda c: popfit[c])
    return pop[winner]

def uniform_crossover(p1, p2):
    c1, c2 = [], []
    for i in range(len(p1)):
        if random.random() < 0.5:
            c1.append(p1[i]); c2.append(p2[i])
        else:
            c1.append(p2[i]); c2.append(p1[i])
    return np.array(c1), np.array(c2)
