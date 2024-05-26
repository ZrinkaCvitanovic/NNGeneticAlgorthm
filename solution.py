import numpy as np
import sys
import copy

features = list()
target = None
training__dataset = list()
testing_dataset = list()
candidate_error = dict()

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
    #https://numpy.org/doc/stable/user/absolute_beginners.html za stvaranje polja
                
class NeuralNetwork:

    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layers = [] #preserve the best solution yet
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html za specifikaciju normalne distribucije
        


    def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

    def forward_propagate(self, X):
        activation = X
        for i in range (len(self.layers)):
            layer = self.layers[i]
            activation = np.dot(activation, layer)
            if i != len(self.layers) - 1:
                activation = self.sigmoid(activation)
        return activation
    
    
    def modify_layers(self, popsize, elitism, p, K, max_iterations):
        current_iterations = 0
        best_solution = None
        best = None
        offspring = list()
        global candidate_error
        
        while len(candidate_error) < popsize:
            candidate = self.generate_chromosome()
            err = self.calculate_error("modify", candidate)
            candidate_error[err] = candidate
            
        while (current_iterations < max_iterations):
            for i in range(2000):
                local_best = None
                parents, everyone = self.select()
                child = self.cross(parents[0], parents[1])
                child = self.mutate(child, p, K)
                err  = self.calculate_error("modify", child)
                everyone[err] = child
                sorted_everyone = list(list(everyone.items()))
                del sorted_everyone[-1]
                total_error = self.calculate_error("train")
                if best is None or total_error < best:
                    best = total_error
                    best_solution = self.layers
                if local_best is None or total_error < local_best:
                        local_best = total_error
            current_iterations += 2000
            print(f"[Train error @{current_iterations}]: {round(local_best, 6)}")
            
        print(f"[Train error @10000]: {round(best, 6)}")
                        
        return None
    
    
    def generate_chromosome(self):
        chromosome = []
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.input_size, self.hidden_layers[0]))) 
        for i in range(1, len(self.hidden_layers)):                              
            chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[i-1], self.hidden_layers[i])))
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[-1], self.output_size)))  
        return chromosome
    
    def select(self):
        first_parent = None
        second_parent = None
        sorted_dict = dict(sorted(candidate_error.items()))
        parents = list(sorted_dict.values())[:2]
        return parents, sorted_dict 
    
    def cross(self, parent1, parent2):
        child = np.array()
        for i in range (len(parent1)):
            child[i] = parent1[i] + parent2[i] / 2.0
        return child
    
    def mutate(self, child, probability, scale):
        for el in child:
            p = np.random.rand()
            if p < probability:
                el += np.random.normal(loc=0.0, scale=scale)
        return child 
            
       

    def calculate_error(self, phase, dataset=None):
        if phase == "train":
            dataset = training__dataset
        elif phase == "train":
            dataset = testing_dataset
        else:
            dataset = dataset
        result = 0.0
        size = 0
        for example in dataset:
            size += 1
            output = self.forward_propagate(example[:-1:1])
            error = np.square(example[-1] - output)
            result += error
        return result / size
    
                        
if __name__ == "__main__":
    file_train = "sine_train.txt"
    file_test = "sine_test.txt"
    nn = [2]
    popsize = 10
    elitism = 1  
    p = 0.1     
    K = 0.1     
    max_iterations = 1000
    read_csv(file_train)
    network = NeuralNetwork(len(features), nn, 1)
    network.modify_layers(popsize, elitism, p, K, max_iterations)
    read_csv(file_test, "test")
    err = network.calculate_error("test")
    print(f"[Test error]: {err}")