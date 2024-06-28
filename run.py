from libs.bareNeuralNets.network import NeuralNetwork
from libs.prettyPrint.printMatrices import PrettyPrint
from libs.mnist.loader import load_data_wrapper
import numpy as np

training_data, validation_data, test_data = load_data_wrapper()

training_data = np.array(list(training_data), dtype=object)
validation_data = np.array(list(validation_data), dtype=object)
test_data = np.array(list(test_data), dtype=object)

training_data, training_labels = training_data[:, 0], training_data[:, 1]
validation_data, validation_labels = validation_data[:, 0], validation_data[:, 1]
test_data, test_labels = test_data[:, 0], test_data[:, 1]

neural_network = NeuralNetwork(structure=[784, 30, 10])
