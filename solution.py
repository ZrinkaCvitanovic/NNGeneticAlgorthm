import numpy as np
import sys
import random
import argparse

features = list()
target = None
training__dataset = list()
testing_dataset = list()

def read_csv(file, purpose="train"):
    first = True
    with open(file, "r", encoding="utf-8") as data:
        for line in data:
            line = line.strip().split(",")
            if first is True:
                for i in range (len(line)): 
                    if i == len(line)-1:
                        target = line[i]
                    else:
                        features.append(line[i])
                first = False
            else:
                if purpose == "train":
                    training__dataset.append(np.array(line, dtype=float))
                else:
                    testing_dataset.append(np.array(line, dtype=float))
                
class NeuralNetwork:

    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layers = [] 
        for i in range (len(self.hidden_layers)):
            self.hidden_layers[i] = int(self.hidden_layers[i])     


    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

    def forward_propagate(self, X, solution):
        activation = X[0:-1]
        for i in range (1, len(solution), 2):
            weight = solution[i-1]
            bias = solution[i]
            activation = np.dot(weight, activation) + bias
            if i < len(solution) - 2:
                activation = self.sigmoid(activation)
        return activation[0]
    
    
    def modify_layers(self, popsize, elitism, p, K, max_iterations):
        current_iterations = 1
        total_lowest_error = None
        optimal_solution = None
        current_lowest_error = None
        population = dict()

        while len(population) < popsize:
                candidate = self.generate_chromosome()
                total_error = 0.0
                for example in training__dataset:
                    err = self.calculate_error_squared(example, candidate)
                    total_error += err
                total_error = total_error / len(training__dataset)
                population[total_error] = candidate
                    
        while (current_iterations <= max_iterations):
            parents = self.select(population)
            child = self.cross(parents[0], parents[1])
            child = self.mutate(child, p, K)
            total_error = 0
            for example in training__dataset:
                err  = self.calculate_error_squared(example, child)
                total_error += err
            total_error = total_error / len(training__dataset)
            if total_error in population.keys():
                continue
            population[total_error] = child
            sorted_population = dict(sorted(population.items()))
            sorted_keys = list(sorted_population.keys())[0:elitism]
            if current_lowest_error is None or sorted_keys[0] < current_lowest_error:
                    current_lowest_error = sorted_keys[0]
            population = dict()
            for item in sorted_keys:
                population[item] = sorted_population[item]
            if current_iterations%2000 == 0:
                print(f"[Train error @{current_iterations}]: {current_lowest_error}")
                if total_lowest_error is None or current_lowest_error < total_lowest_error:
                    total_lowest_error = current_lowest_error
                    optimal_solution = population[sorted_keys[0]]
                
            while len(population) < popsize:
                candidate = self.generate_chromosome()
                total_error = 0.0
                for example in training__dataset:
                    err = self.calculate_error_squared(example, candidate)
                    total_error += err
                total_error = total_error / len(training__dataset)
                population[total_error] = candidate
            current_iterations += 1
                    
        print(f"[Train error @{max_iterations}]: {total_lowest_error}")       
        return optimal_solution 
    
    
    def generate_chromosome(self):
        chromosome = []
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[0], self.input_size))) 
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[0]))) 
        for i in range(1, len(self.hidden_layers)):                              
            chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[i], self.hidden_layers[i-1])))
            chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[i])))
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.output_size, self.hidden_layers[-1])))
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.output_size)))  
        return chromosome
    
    def select(self, population):
        indexes = set()
        while len(indexes) < 3:
            indexes.add(np.random.randint(len(population)))
        keys = list(population.keys())
        temp = dict()
        for i in indexes:
            temp[keys[i]] = population[keys[i]]
        sorted_dict = dict(sorted(temp.items()))
        parents = list(sorted_dict.values())[:2]
        return parents 
    
    def cross(self, parent1, parent2):
        child = []
        for matrix1, matrix2 in zip(parent1, parent2):
            temp = np.add(matrix1, matrix2)
            temp = np.divide(temp, 2)
            child.append(temp)
        return child
    
    def mutate(self, child, probability, scale):
        for i in range (len(child)):
            p = np.random.rand()
            if p < probability:
                number = np.random.normal(loc=0.0, scale=scale)
                child[i] = np.add(child[i], number)
        return child 

    def calculate_error_squared(self, example, solution):
        output = self.forward_propagate(example, solution)
        return pow((example[-1] - output), 2)
    
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser() #prvi labos
    parser.add_argument( '--train', help='Path to a file with training data.' )
    parser.add_argument( '--test', help='Path to a file with testing data.' )
    parser.add_argument( '--nn', help='Neural network architecture. Separate hidden layers with "s".' )
    parser.add_argument( '--popsize', help='Population size.' )
    parser.add_argument( '--elitism', help='Define how many best candidates are preserved (if none, enter 0).')
    parser.add_argument( '--p', help='Probability of a mutation' )
    parser.add_argument( '--K', help='Standard deviation used for mutation a chromosome' )
    parser.add_argument( '--iter', help='Maximum number of iterations')
    arguments = parser.parse_args()
    file_train = arguments.train
    file_test = arguments.test
    nn = arguments.nn.split("s")[:-1]
    popsize = int(arguments.popsize)
    elitism = int(arguments.elitism)
    p = float(arguments.p)
    K = float(arguments.K)
    max_iterations = int (arguments.iter)
    read_csv(file_train)
    network = NeuralNetwork(len(features), nn, 1)
    optimal_solution = network.modify_layers(popsize, elitism, p, K, max_iterations)
    read_csv(file_test, "test")
    total_error = 0.0
    for example in testing_dataset:
        err = network.calculate_error_squared(example, optimal_solution)
        total_error += err
    total_error = total_error / len(testing_dataset)
    print(f"[Test error]: {total_error}")