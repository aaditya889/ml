import numpy as np
from libs.prettyPrint.printMatrices import PrettyPrint
import random

  
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
class NNParamaters:
  class Connectivity:
    fully_connected = 'fully_connected'

  class ActivationFunctions:
    def sigmoid(x):
      return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
      return  sigmoid(x) * (1 - sigmoid(x))
    
    def relu(x):
      return np.maximum(0, x)
    
    def relu_derivative(x):
      y = (x > 0) * 1
      return y
    
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
               learning_rate=4,
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
        self.weights = [np.random.randn(structure[i+1], structure[i]) for i in range(self.layers-1)]
        self.biases = [np.random.randn(structure[i+1], 1) for i in range(self.layers-1)]
      case _:
        raise ValueError(f"Connectivity {connectivity} not supported")

  def feedforward(self, input_data):
    zs = [input_data]
    activations = [input_data]
    for i in range(self.layers-1):
      z = np.matmul(self.weights[i], activations[i]) + self.biases[i]
      zs.append(z)
      activations.append(self.activation_function(z))
    
    return zs, activations
  
  # TODO: Implement a non-recursive backpropagation algorithm.
  def backpropagation(self, learning_constant, zs, activations, current_delta, current_layer, delta_ws, delta_bs):
    if current_layer == 0:
      return

    delta_w = learning_constant * np.matmul(current_delta, activations[current_layer-1].T)
    delta_b = learning_constant * current_delta
    # PrettyPrint.matrix(delta_w)
    
    delta_ws[current_layer-1] += delta_w
    delta_bs[current_layer-1] += delta_b
    previous_delta = np.matmul(self.weights[current_layer-1].T, current_delta) * self.activation_function_derivative(zs[current_layer-1])
    self.backpropagation(learning_constant, zs, activations, previous_delta, current_layer-1, delta_ws, delta_bs)

  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """Train the neural network using mini-batch stochastic
    gradient descent.  The ``training_data`` is a list of tuples
    ``(x, y)`` representing the training inputs and the desired
    outputs.  The other non-optional parameters are
    self-explanatory.  If ``test_data`` is provided then the
    network will be evaluated against the test data after each
    epoch, and partial progress printed out.  This is useful for
    tracking progress, but slows things down substantially."""
    if test_data: 
      n_test = len(test_data)
    
    n = len(training_data)
    
    correct_results = self.evaluate(test_data)
    print(f"Without training: {correct_results} / {n_test} :: Percentage = {100*correct_results / n_test}%")

    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.train_minibatch(mini_batch, learning_rate=eta)
      if test_data:
        correct_results = self.evaluate(test_data)
        print(f"Epoch {j}: {correct_results} / {n_test} :: Percentage = {100*correct_results / n_test}%")
      else:
        print("Epoch {0} complete".format(j))


  def train_minibatch(self, training_data, learning_rate=None):
    if not learning_rate:
      learning_rate = self.learning_rate
    
    n = len(training_data)
    learning_constant = learning_rate / n
    
    delta_ws = [np.zeros(w.shape) for w in self.weights]
    delta_bs = [np.zeros(b.shape) for b in self.biases]

    for datum, label in training_data:
      if (len(datum.shape) == 1):
        curr_datum = datum.reshape(datum.shape+(1,))
        curr_label = label.reshape(label.shape+(1,))
      else:
        curr_datum = datum
        curr_label = label

      zs, activations = self.feedforward(curr_datum)    
      last_delta = self.cost_function_derivative(activations[-1], curr_label) * self.activation_function_derivative(zs[-1])
      self.backpropagation(learning_constant, zs, activations, last_delta, self.layers-1, delta_ws, delta_bs)
        
    for i in range(self.layers-1):
      # print(f'Updating weights for layer {i}')
      # PrettyPrint.matrix(delta_ws[i])
      self.weights[i] -= delta_ws[i]
      # print(f'Updating biases for layer {i}')
      # PrettyPrint.matrix(delta_bs[i])
      self.biases[i] -= delta_bs[i]


  def predict(self, data):
    zs, activations = self.feedforward(data)
    return activations[-1]
  
  def evaluate(self, test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    # for (x, y) in test_data:
    #   print(f'Predicting for data of shape: {x.shape}')
    #   # PrettyPrint.matrix(x)
    #   self.feedforward(x)
    test_results = [(np.argmax(self.feedforward(x)[1][-1]), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)