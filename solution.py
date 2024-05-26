import numpy as np
import sys

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

    def __init__(self, input_size, hidden_layers, output_size): #hidden layers je lista koliko imam layera
        self.layers = []
        
        #https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html za specifikaciju normalne distribucije
        self.layers.append(np.random.normal(loc=0.0, scale=0.01, size=(input_size, hidden_layers[0]))) 
        for i in range(1, len(hidden_layers)):                              
            self.layers.append(np.random.normal(loc=0.0, scale=0.01, size=(hidden_layers[i-1], hidden_layers[i])))
        self.layers.append(np.random.normal(loc=0.0, scale=0.01, size=(hidden_layers[-1], output_size)))  

    
    def forward_propagate(self, X):
        activation = X
        for i in range (len(self.layers)):
            layer = self.layers[i]
            activation = np.dot(activation, layer)
            if i != len(self.layers) - 1:
                activation = self.sigmoid(activation)
        return activation
                       
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def calculate_error():
        result = 0.0
        size = 0
        for example in training__dataset:
            size += 1
            output = network.forward_propagate(example[:-1:1])
            error = numpy.square(training__dataset[-1] - output[0])
            result += error
        return result / size

    def modify_layers(popsize, elitism, p, K):
        return None
                 
                
    
    
if __name__ == "__main__":
    file_train = "sine_train.txt"
    file_test = "sine_test.txt"
    nn = [5]
    popsize = 10 #veličlia populacije
    elitism = 1  #je li elitistički
    p = 0.1     #vjerojatnost mutacije
    K = 0.1     #skala mutacije
    max_iterations = 1000
    current_iterations = 0
    read_csv(file_train)
    network = NeuralNetwork(len(features), nn, 1)
    while (current_iterations <= max_iterations ):
        for i in range(2000):
            network.modify_layers(popsize, elitism, p, K)
            #change layers for 2000 times
        err = network.calculate_error()
        print(f"[Train error @{current_iterations + 2000}]: {err}")
        current_iterations += 2000

    read_csv(file_test)
    current_iterations = 0
    err = network.calculate_error()
    print(f"[Test error]: {err}")
    
