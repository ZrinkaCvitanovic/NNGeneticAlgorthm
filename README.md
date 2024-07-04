# Training a neural network using genetic algorithm

This is a simple program used to demonstrate how training a neural network and genetic algoritm work. 

## Technical details
All data is written in a .csv file, the last column always being the output. The size of the output is always 1. The neural network is represented with a class in order to avoid using global variables.

## Algorithm flow
All the hyperparameters are specified as command line arguments. Each chromosome is an array of matrices and each matrix is either weight matrix or bias matrix for each layer of a neural network. The initial values are generated randomly using normal distribution. The number of solutions generated is specified by "popsize" argument. For each of the solutions (chromosomes) the program calculates mean square error. The chromosomes are sorted from the least to greatest error.  


**Selection of parents** is implemented using the tournament selection. Three random candidates are extracted and the two with lowest errors are selected to be parents. Then those two are crossed.  
The **cross operation** is implemented as an arithmetic mean of each element in the matrix. Afterwards it is possible that the child is mutated.  
The probability of that **mutation** is specified by the "p" argument and the mutation is implemented by adding a random number generated using normal distribution with scale "K". The number is added to every element of the matrix. 


In the end the program calculates the mean square error of the newly generated chromosome as well and sorts the entire population by the mean square error. The program supports elitistic algorithms which means several of the best candidates can be perserved for the next generation. The number of preserved chromosomes is defined with the "elitism" parameter. The algorithm is performed for the "maxiter" number of times. The current best solution is preserved in the class object "layers". In the end, the best solution is selected and its error is calulcated on the new dataset - testing dataset. 

## Examples of specifying arguments
*python solution.py --train train.txt --test test.txt --nn 5s --popsize 10 --elitism 1 --p 0.2 --K 0.1 --iter 1000*    
   * training data set is in file train.txt and testing dataset in test.txt  
   * the neural network has one hidden layer of 5 neurons
     * output size is always 1 and the input size is calculated from the training dataset
     * input size is number of columns minus one, since the last column is always the expected result  
   * for each generation there are 10 chromosomes  
   * the one best chromosome is preserved for the next generation  
   * the probability of the mutation is 0.2 and the scale of mutation is 0.1  
   * the maximum number of iterations is 1000  

*python solution.py --train train.txt --test test.txt --nn 5s10s --popsize 100 --elitism 5 --p 0.5 --K 0.1 --iter 1000000*    
   * training data set is in file train.txt and testing dataset in test.txt   
   * the neural network has two hidden layers: one has 5 and the other 10 neurons  
   * for each generation there are a 100 chromosomes  
   * five best chromosomes are preserved for the next generation  
   * the probability of the mutation is 0.5 and the scale of mutation is 0.1  
   * the maximum number of iterations is 1000000  
