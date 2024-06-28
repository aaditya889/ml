import numpy as np
from libs.prettyPrint.printMatrices import PrettyPrint

VERBOSE_DEBUG = False
NON_VERBOSE_DEBUG = False
NON_VERBOSE_DEBUG_2 = False
  
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
        self.weights = [np.random.rand(structure[i+1], structure[i]) for i in range(self.layers-1)]
        self.biases = [np.random.rand(structure[i+1], 1) for i in range(self.layers-1)]
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
  
  def backpropagation(self, learning_constant, zs, activations, current_delta, current_layer, delta_ws, delta_bs):
    if current_layer == 0:
      return

    delta_w = learning_constant * np.matmul(current_delta, activations[current_layer-1].T)
    delta_b = learning_constant * current_delta
    delta_ws[current_layer-1] += delta_w
    delta_bs[current_layer-1] += delta_b
    previous_delta = np.matmul(self.weights[current_layer-1].T, current_delta) * self.activation_function_derivative(zs[current_layer-1])
    self.backpropagation(learning_constant, zs, activations, previous_delta, current_layer-1, delta_ws, delta_bs)

  
  def train_minibatch(self, training_data, labels, epochs=1000):
    n = len(training_data)
    learning_constant = self.learning_rate / n

    while epochs > 0:
      delta_ws = [np.zeros(w.shape) for w in self.weights]
      delta_bs = [np.zeros(b.shape) for b in self.biases]

      for datum, label in zip(training_data, labels):
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
        self.weights[i] -= delta_ws[i]
        self.biases[i] -= delta_bs[i]

      epochs -= 1

  def predict(self, data):
    if NON_VERBOSE_DEBUG:
      print(f'Predicting for data of shape: {data.shape}')
      PrettyPrint.matrix(data)
    zs, activations = self.feedforward(data)
    return activations[-1]
  
  def evaluate(self, test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(np.argmax(self.feedforward(x)), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)