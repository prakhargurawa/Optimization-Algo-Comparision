# Optimization-Algo-Comparision
Different optimization algorithms like Hill climbing, Simulated annealing,  Late accepted Hill climbing , Genetic Algorithm is implemented from scratch.

For algorithms like Covariance Matrix adaptation and Particle Swarm Algorithm external python libraries are used.

Finally a comparision of algorithms are presented where each algo ran five times on random seed and mean and standard deviation of objective function is presented.

The problem considered here is facility location problem, where we have n number of tree , each represented using (x,y) coordinates and number  of fruits on each tree. This information is present in "fruit_picking.txt" as x_coordinate , y_coordinate and num_of_fruits.  We need to place m bins so that the amount of movement is mimimum. Here we need to place m = 6 bins , so total 12 real value decision variable. 

Objective Function

Minimize f(x1,x2,....,x2m) = Σi wi minj(d(ti,bj))

Main file : OptimizationAlgoCompare.py

Report : OptimizationAlgoComparision.pdf

	

