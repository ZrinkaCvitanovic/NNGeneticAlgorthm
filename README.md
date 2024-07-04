# Training a neural network using genetic algorithm

This is a simple program used to demonstrate how training a neural network and genetic algoritm work. 

## Technical details
All data is written in a .csv file, the last column always being the output. The size of the output is always 1. The neural network is represented with a class in order to avoid using global variables.

## Algorithm flow
All the hyperparameters are specified as command line arguments. Each chromosome is an array of matrices and each matrix is either weight matrix or bias matrix for each layer of a neural network. The initial values are generated randomly using normal distribution. The number of solutions generated is specified by "popsize" argument. For each of the solutions (chromosomes) the program calculates mean square error. The chromosomes are sorted from the least to greatest error. Parents of a new chromosome are selected using the tournament selection. Three random candidates are extracted and the two with lowest errors are selected to be parents. Then those two are crossed. The cross operation is implemented as an arithmetic mean of each element in the matrix. Afterwards it is possible that the child is mutated. The probability of that mutation is specified by the "p" argument. The program calculates the mean square error of the newly generated chromosome as well and sorts the entire population by the mean square error. The program supports elitistic algorithms which means several of the best candidates can be perserved for the next generation. The number of preserved chromosomes is defined with the "elitism" parameter. The algorithm is performed for the "maxiter" number of times. The current best solution is preserved in the class object "layers". In the end, the best solution is selected and its error is calulcated on the new dataset - testing dataset. 
