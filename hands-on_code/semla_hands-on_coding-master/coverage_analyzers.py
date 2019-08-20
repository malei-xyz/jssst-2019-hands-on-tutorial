import numpy as np
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
import itertools
import pickle
import os

class NC():

    def __init__(self, layer_names, threshold=0.25):
        '''
        Initialize the model to be tested
        :param threshold: threshold to determine if the neuron is activated
        :param layer_names: Only these layers are considered for neuron coverage
        '''
        self.threshold = threshold
        # the layers that are considered in neuron coverage computation
        self.layer_names = layer_names
        self.tensors = []
        # init coverage table
        self.neurons_data = {}

        for layer_name in self.layer_names:
            layer_tensor = _as_graph_element(layer_name)
            self.tensors.append(layer_tensor)
            dims = layer_tensor.get_shape()[1:]
            neuron_indexes = []
            for dim in dims:
                neuron_indexes.append([i for i in range(dim)]) 
            for neuron_idx in itertools.product(*neuron_indexes):
                self.neurons_data[(layer_name, neuron_idx)] = False

    def get_tensors(self):
        '''
        get the references to the graph elements that represent the layers activationss
        :return: activations' tensors
        '''
        return self.tensors

    def update_coverage(self, tensors_values):
        '''
        update the coverage of neurons after execution
        :param tensors_values: the layers' activations (tensors)
        :return:
        '''
        for i in range(len(self.layer_names)):
            self.update_neurons_coverage(self.layer_names[i], tensors_values[i])

    def scale(self, layer_outputs, rmax=1, rmin=0):
        '''
        scale the intermediate layer's output between 0 and 1
        :param layer_output: the layer's output tensor
        :param rmax: the upper bound of scale
        :param rmin: the lower bound of scale
        :return:
        '''
        divider = (layer_outputs.max() - layer_outputs.min())
        if divider == 0:
            return np.zeros(shape=layer_outputs.shape)
        X_std = (layer_outputs - layer_outputs.min()) / divider
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled

    def update_neurons_coverage(self, layer_name, layer_output):
        '''
        update the coverage of neurons belong to a layer
        :param layer_name: the layer's name
        :param layer_output: the layer's output tensor
        :return:
        '''
        layer_neuron_indexes = [neuron_idx for (l_name, neuron_idx),v in self.neurons_data.items() if l_name == layer_name and not v]
        scaled = self.scale(layer_output)
        for neuron_idx in layer_neuron_indexes:
            idx = (slice(None,None),*neuron_idx)
            neuron_out = scaled[idx]
            if np.max(neuron_out) > self.threshold:
                self.neurons_data[(layer_name, neuron_idx)] = True

    def curr_coverage(self):
        '''
        compute the overall neurons coverage
        :return: the number of covered neurons, the total number of neurons, ratio of covered neurons
        '''
        covered_neurons = [1 for k,v in self.neurons_data.items() if v]
        covered_neurons_count = np.sum(covered_neurons)
        total_neurons_count = len(self.neurons_data.items())
        return covered_neurons_count, total_neurons_count, covered_neurons_count/float(total_neurons_count)    

class KMNC():

    def __init__(self, layer_names, data_path, k=100):
        '''
        Initialize the model to be tested
        :param k: the number of sections to cover for each neuron
        :param layer_names: Only these layers are considered for neuron coverage
        :param data_path: the path of binary data containing neurons' bounds information
        '''
        self.data_path = data_path 
        # the layers that are considered in neuron coverage computation
        self.layer_names = layer_names

        for layer_name in self.layer_names:
            layer_tensor = _as_graph_element(layer_name)
            self.tensors.append(layer_tensor)

    def neurons_data_exist(self):
        return os.path.isfile(self.data_path)
    # to complete
    def save_neurons_data(self):
        with open(self.data_path, 'wb') as data_file:
            # use pickle.dump(obj, data_file)
            pass
    # to complete
    def load_neurons_data(self):
        with open(self.data_path, 'rb') as data_file:
            # use obj = pickle.load(data_file)
            pass

    def get_tensors(self):
        '''
        get the references to the graph elements that represent the layers activationss
        :return: activations' tensors
        '''
        return self.tensors

    def update_bounds(self, tensors_values):
        '''
        update the bounds of neurons after execution
        :param tensors_values: the layers' activations (tensors)
        :return:
        '''
        for i in range(len(self.layer_names)):
            self.update_neurons_bounds(self.layer_names[i], tensors_values[i])

    def update_coverage(self, tensors_values):
        '''
        update the coverage of neurons after execution
        :param tensors_values: the layers' activations (tensors)
        :return:
        '''
        for i in range(len(self.layer_names)):
            self.update_neurons_sections_coverage(self.layer_names[i], tensors_values[i])
    # to complete
    def update_neurons_bounds(self, layer_name, layer_output):
        '''
        update the empirical boundaries of neurons belong to a layer
        :param layer_name: the layer's name
        :param layer_output: the layer's output tensor
        :return:
        '''
        pass
    # to complete
    def update_neurons_sections_coverage(self, layer_name, layer_output):
        '''
        update the coverage of neurons' sections belong to a layer
        :param layer_name: the layer's name
        :param layer_output: the layer's output tensor
        :return:
        '''
        pass
    # to complete
    def curr_coverage(self):
        '''
        compute the overall neurons' sections coverage
        :return: the number of covered neurons' sections, the total number of neurons' sections, ratio of covered neurons' sections
        '''   
        pass