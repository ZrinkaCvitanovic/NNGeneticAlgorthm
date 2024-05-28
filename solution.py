import numpy as np
import sys
import copy

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

    def forward_propagate(self, X, solution):
        activation = X[0:-1]
        for i in range (1, len(solution), 2):
            weight = solution[i-1]
            bias = solution[i]
            activation = np.dot(weight, activation) + bias
            if i != len(self.layers) - 1:
                activation = self.sigmoid(activation)
        return activation[0]
    
    
    def modify_layers(self, popsize, elitism, p, K, max_iterations):
        current_iterations = 1
        total_lowest_error = None
        optimal_solution = None
        candidate_error = dict()
        current_best_candidate = None
        current_lowest_error = None
                    
        while (current_iterations <= max_iterations):
            
            while len(candidate_error) < popsize:
                candidate = self.generate_chromosome()
                total_error = 0.0
                for example in training__dataset:
                    err = self.calculate_error_squared(example, candidate)
                    total_error += err
                total_error = total_error / len(training__dataset)
                candidate_error[total_error] = candidate
        
            parents = self.select(candidate_error)
            child = self.cross(parents[0], parents[1])
            child = self.mutate(child, p, K)
            total_error = 0
            
            for example in training__dataset:
                err  = self.calculate_error_squared(example, child)
                total_error += err
                
            total_error = total_error / len(training__dataset)
            candidate_error[total_error] = child
            sorted_everyone = dict(sorted(candidate_error.items()))
            iter_lowest_error = list(sorted_everyone.keys())[0] #https://stackoverflow.com/questions/30362391/how-do-you-find-the-first-key-in-a-dictionary
            if current_lowest_error is None or iter_lowest_error < current_lowest_error:
                    current_lowest_error = iter_lowest_error
                    current_best_candidate = candidate_error[iter_lowest_error]
            if total_lowest_error is None or current_lowest_error < total_lowest_error:
                    total_lowest_error = current_lowest_error
                    optimal_solution = current_best_candidate
            candidate_error = dict()
            candidate_error[current_lowest_error] = current_best_candidate #elitism is 1
            if current_iterations%2000 == 0:
                print(f"[Train error @{current_iterations}]: {round(current_lowest_error, 6)}")
                current_best_candidate = None 
                current_lowest_error = None 
            current_iterations += 1
                    
        print(f"[Train error @10000]: {round(total_lowest_error, 6)}")       
        self.layers = optimal_solution 
        return None
    
    
    def generate_chromosome(self):
        chromosome = []
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[0], self.input_size))) 
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[0]))) 
        for i in range(1, len(self.hidden_layers)):                              
            chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[i], self.hidden_layers[i-1])))
            chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.hidden_layers[i])))
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.output_size, self.hidden_layers[0])))
        chromosome.append(np.random.normal(loc=0.0, scale=0.01, size=(self.output_size)))  
        return chromosome
    
    def select(self, candidate_error):
        sorted_dict = dict(sorted(candidate_error.items()))
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
        for el in child:
            for e in el:
                p = np.random.rand() #https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
                if p < probability:
                    number = np.random.normal(loc=0.0, scale=scale)
                    e = np.add(e, number)
        return child 

    def calculate_error_squared(self, example, solution):
        output = self.forward_propagate(example, solution)
        return pow((example[-1] - output), 2)
    
                        
if __name__ == "__main__":
    file_train = "sine_train.txt"
    file_test = "sine_test.txt"
    nn = [2]
    popsize = 10
    elitism = 1  
    p = 0.1     
    K = 0.1     
    max_iterations = 10000
    read_csv(file_train)
    network = NeuralNetwork(len(features), nn, 1)
    network.modify_layers(popsize, elitism, p, K, max_iterations)
    read_csv(file_test, "test")
    total_error = 0.0
    for example in testing_dataset:
        err = network.calculate_error_squared(example, network.layers)
        total_error += err
    total_error = total_error / len(testing_dataset)
    print(f"[Test error]: {total_error}")