from libs.bareNeuralNets.network import NeuralNetwork
from libs.prettyPrint.printMatrices import PrettyPrint
from libs.mnist.loader import load_data_wrapper
import numpy as np

# training_data = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
# training_labels = np.array([[1, 1, 0, 0]]).T
training_data, validation_data, test_data = load_data_wrapper()
# (data, training_labels) = list(training_data)
training_data = np.array(list(training_data), dtype=object)
validation_data = np.array(list(validation_data), dtype=object)
test_data = np.array(list(test_data), dtype=object)

training_data, training_labels = training_data[:, 0], training_data[:, 1]
validation_data, validation_labels = validation_data[:, 0], validation_data[:, 1]
test_data, test_labels = test_data[:, 0], test_data[:, 1]

print(test_labels)
# print(data[0][0].shape)
# print(training_data[0][1])
# print(training_labels.shape)
# PrettyPrint.matrix(training_data)
# PrettyPrint.matrix(training_labels)

# input_data = np.array([1, 1]).reshape(2, 1)
# neural_network = NeuralNetwork([2, 3, 1])

# for i, w in enumerate(neural_network.weights):
#   print(f'weight #{i}')
#   PrettyPrint.matrix(w)
  
# for i, b in enumerate(neural_network.biases):
#   print(f'bias #{i}')
#   PrettyPrint.matrix(b)

# print('Initial Predictions:')
# print('0, 0')
# PrettyPrint.matrix(neural_network.predict(np.array([0, 0]).reshape(2, 1)), print_name=False)
# print('0, 1')
# PrettyPrint.matrix(neural_network.predict(np.array([0, 1]).reshape(2, 1)), print_name=False)
# print('1, 0')
# PrettyPrint.matrix(neural_network.predict(np.array([1, 0]).reshape(2, 1)), print_name=False)
# print('1, 1')
# PrettyPrint.matrix(neural_network.predict(np.array([1, 1]).reshape(2, 1)), print_name=False)

# # for i in range(1):
# neural_network.train_minibatch(training_data, training_labels, epochs=1000)

# print('Post Training Predictions:')
# print('0, 0')
# PrettyPrint.matrix(neural_network.predict(np.array([0, 0]).reshape(2, 1)), print_name=False)
# print('0, 1')
# PrettyPrint.matrix(neural_network.predict(np.array([0, 1]).reshape(2, 1)), print_name=False)
# print('1, 0')
# PrettyPrint.matrix(neural_network.predict(np.array([1, 0]).reshape(2, 1)), print_name=False)
# print('1, 1')
# PrettyPrint.matrix(neural_network.predict(np.array([1, 1]).reshape(2, 1)), print_name=False)

# # zs, activations = neural_network.feedforward(input_data)

# PrettyPrint.matrix(input_data)

# # for i, a in enumerate(activations):
# #   print(f'activation #{i} ', end='')
# #   PrettyPrint.matrix(a)
