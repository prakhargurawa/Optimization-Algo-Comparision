# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 20:27:07 2020

@author: prakh
"""
# LIBRARY IMPORT
import numpy as np


"""Late-acceptance hill-climbing (LAHC) is a stochastic hill-climbing
algorithm with a history mechanism, proposed by Bykov
[http://www.cs.nott.ac.uk/~yxb/LAHC/LAHC-TR.pdf].

The history mechanism is very simple but in some domains it seems to
provide a remarkable performance improvement compared to hill-climbing
itself and other heuristics. The big advantage is its simplicity:
fewer parameters to tune compared to many alternative methods.

In standard stochastic hill-climbing, we accept a move to a new
proposed point (created by mutation) if that point is as good as or
better than the current point.

In LAHC, we accept the move if the new point is as good as or better
than that we encountered L steps ago. L is the only new parameter
compared to hill-climbing: it stands for history length.

"""


"""
This is the LAHC pseudo-code from Bykov and Burke.

Produce an initial solution s
Calculate initial cost function C(s)
Specify Lfa
For all k in {0...Lfa-1} f_k := C(s)
First iteration I=0;
Do until a chosen stopping condition
    Construct a candidate solution s*
    Calculate its cost function C(s*)
    v := I mod Lfa
    If C(s*)<=fv or C(s*)<=C(s)
    Then accept the candidate (s:=s*)
    Else reject the candidate (s:=s)
    Insert the current cost into the fitness array fv:=C(s)
    Increment the iteration number I:=I+1

"""

# L is history length
# n is the budget of evaluations
# C is the cost function
# init is a function that creates an initial individual
# nbr is a neighbourhood function

def LAHC(L, n, C, init, nbr):
    s = init()            # initial solution
    Cs = C(s)             # cost of current solution
    best = s              # best-ever solution
    Cbest = Cs            # cost of best-ever
    f = [Cs] * L          # initial history
    history = []          # create history
    # print(0, Cbest, best)
    for I in range(1, n): # number of iterations
        s_ = nbr(s)       # candidate solution
        Cs_ = C(s_)       # cost of candidate
        if Cs_ < Cbest:   # minimising
            best = s_     # update best-ever
            Cbest = Cs_
            history.append((I, Cbest)) # save history
            # print(I, Cbest, best)
        v = I % L         # v indexes I circularly
        if Cs_ <= f[v] or Cs_ <= Cs:
            s = s_        # accept candidate
            Cs = Cs_      # (otherwise reject)
        f[v] = Cs         # update circular history
    return best, Cbest,np.array(history)