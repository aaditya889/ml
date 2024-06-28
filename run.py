from libs.bareNeuralNets.network import NeuralNetwork
from libs.prettyPrint.printMatrices import PrettyPrint
import numpy as np

# test_data = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
# test_labels = np.array([[1, 1, 0, 0]]).T

input_data = np.array([1, 1]).reshape(2, 1)
neural_network = NeuralNetwork([2, 2, 1])
zs, activations = neural_network.feedforward(input_data)

PrettyPrint.matrix(input_data)

for i, w in enumerate(neural_network.weights):
  print(f'weight #{i} ', end='')
  PrettyPrint.matrix(w)
  
for i, b in enumerate(neural_network.biases):
  print(f'bias #{i} ', end='')
  PrettyPrint.matrix(b)

for i, a in enumerate(activations):
  print(f'activation #{i} ', end='')
  PrettyPrint.matrix(a)
  
# PrettyPrint.matrix(test_data)
# PrettyPrint.matrix(test_labels)  
# print(PrettyPrint().move_cursor(0, -4))
# print('xxxxxxxxxxxxxx')
