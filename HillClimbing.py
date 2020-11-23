# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:56:20 2020

@author: prakh
"""
# LIBRARY IMPORT
import numpy as np

# the simplest metaheuristic search algorithm
def HC(f, init, nbr, its, stop=None):
    """
    f: objective function X -> R (where X is the search space)
    init: function giving random element of X
    nbr: function X -> X which gives a neighbour of the input x
    its: number of iterations, ie fitness evaluation budget
    stop: termination criterion (X, R) -> bool
    return: best ever x
    
    In this version, we store and return a history of
    best objective values; we avoid wasting objective evaluations;
    we allow a termination criterion.
    """
    history = [] # create history
    x = init()
    fx = f(x) # fx stores f of current best point
    for i in range(its):
        xnew = nbr(x)
        fxnew = f(xnew) # avoid re-calculating f
        if fxnew < fx: 
            x = xnew
            fx = fxnew
        history.append((i, fx)) # save history
        if stop is not None and stop(x, fx): # a termination condition
            break
    return x, np.array(history) # return history