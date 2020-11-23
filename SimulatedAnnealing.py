# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:19:28 2020

@author: prakh
"""
# LIBRARY IMPORT
import math
import numpy as np
import random


def SA(f, init, nbr, T, alpha, maxits):
    """Simulated annealing. Assume we are minimising.
    Return the best ever x and its f-value.

    Pass in initial temperature T and decay factor alpha.

    T decays by T *= alpha at each step.
    """
    x = init() # generate an initial random solution
    fx = f(x)
    bestx = x
    bestfx = fx
    history = [] # create history

    for i in range(1, maxits):
        xnew = nbr(x) # generate a neighbour of x
        fxnew = f(xnew)
        
        # "accept" xnew if it is better, OR even if worse, with a
        # small probability related to *how much worse* it is. assume
        # we are minimising, not maximising.
        if fxnew < fx or random.random() < math.exp((fx - fxnew) / T):
            x = xnew
            fx = fxnew

            # make sure we keep the best ever x also
            if fxnew < bestfx:
                bestx = x
                bestfx = fx
            
        T *= alpha # temperature decreases
        #print(i, fx, T)
        history.append((i, fx)) # save history
    return bestx, bestfx,np.array(history)