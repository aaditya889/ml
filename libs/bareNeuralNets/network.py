import numpy as np
from libs.prettyPrint.printMatrices import PrettyPrint

class NNParamaters:
  class Connectivity:
    fully_connected = 'fully_connected'
  
  class ActivationFunctions:
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
      return x * (1 - x)
    
    def relu(x):
      return np.maximum(0, x)
    
    def relu_derivative(x):
      return 1 if x > 0 else 0
    
  class LossFunctions:
    def mean_squared_error(activations, labels):
      return np.mean((labels - activations) ** 2)
    
    def mean_squared_error_derivative(activations, labels):
      return activations - labels
    
    # def cross_entropy(y_true, y_pred):
    #   return -np.sum(y_true * np.log(y_pred))
    
    # def cross_entropy_derivative(y_true, y_pred):
    #   return y_pred - y_true
  

class NeuralNetwork:
  def __init__(self, structure: list=[], connectivity: str=NNParamaters.Connectivity.fully_connected, 
               learning_rate=0.5,
               activation_function=NNParamaters.ActivationFunctions.sigmoid,
               activation_function_derivative=NNParamaters.ActivationFunctions.sigmoid_derivative,
               cost_function_derivative=NNParamaters.LossFunctions.mean_squared_error_derivative):

    self.structure = structure
    self.layers = len(structure)
    self.connectivity = connectivity
    self.learning_rate = learning_rate
    self.activation_function = activation_function
    self.cost_function_derivative = cost_function_derivative
    self.activation_function_derivative = activation_function_derivative
    
    match connectivity:
      case NNParamaters.Connectivity.fully_connected:
        self.weights = [np.random.rand(structure[i+1], structure[i]) for i in range(self.layers-1)]
        self.biases = [np.random.rand(structure[i+1], 1) for i in range(self.layers-1)]
      case _:
        raise ValueError(f"Connectivity {connectivity} not supported")

  def feedforward(self, input_data):
    zs = []
    activations = [input_data]
    for i in range(self.layers-1):
      z = np.matmul(self.weights[i], activations[i]) + self.biases[i]
      zs.append(z)
      activations.append(self.activation_function(z))
    
    return zs, activations
  
  def backpropagation(self, learning_constant, activations, current_delta, current_layer, delta_ws, delta_bs):
    if current_layer == 0:
      return
    
    delta_ws[current_layer] += learning_constant * np.matmul(current_delta, activations[current_layer-1].T)
    delta_bs[current_layer] += learning_constant * current_delta
    
    previous_delta = np.matmul(self.weights[current_layer-1].T, current_delta) * self.activation_function_derivative(activations[current_layer-1])
    self.backpropagation(learning_constant, activations, previous_delta, current_layer-1, delta_ws, delta_bs)

  
  def train_minibatch(self, training_data, labels):
    n = len(training_data)
    learning_constant = self.learning_rate / n
    delta_ws = [np.zeros(w.shape) for w in self.weights]
    delta_bs = [np.zeros(b.shape) for b in self.biases]
    
    for datum, label in zip(training_data, labels):
      zs, activations = self.feedforward(datum)
      last_delta = self.cost_function_derivative(activations[-1], label) * self.activation_function_derivative(activations[-1])
      self.backpropagation(learning_constant, activations, last_delta, self.layers-1, delta_ws, delta_bs)

    for i in range(self.layers-1):
      self.weights[i] -= delta_ws[i]
      self.biases[i] -= delta_bs[i]


  def predict(self, data):
    # Implement the prediction logic here
    pass
  
  def evaluate(self, data, labels):
    # Implement the evaluation logic here
    pass