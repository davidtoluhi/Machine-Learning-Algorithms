import numpy 
import math

class EmbeddedLayer(object): 


    def __init__(self, sparse_input_length, regular_input_length, target_low_dimension, total_neurons_in_layer): 

        self.target_dimension = target_low_dimension
        self.sparse_input_length = sparse_input_length
        self.neurons = []
        self.total_neurons_in_layer = total_neurons_in_layer
        self.layer_outputs = []

        self.sparse_neurons = numpy.matrix( 
                numpy.random.random(
                    (target_low_dimension, # number of neurons in this layer
                    sparse_input_length+1) # weights for the neurons in this layer: number of neurons in previous layer plus a bias 
                )
            )
        self.regular_neurons = numpy.matrix( 
                numpy.random.random(
                    (total_neurons_in_layer-target_low_dimension, # number of neurons in this layer
                    regular_input_length+1) # weights for the neurons in this layer: number of neurons in previous layer plus a bias 
                )
            )
        
    def feedForward(self, sparse_input, regular_input):
        print(sparse_input)
        print(regular_input)
        baised_sparsed_input = numpy.c_[sparse_input, 1] # add a 1 to the input for the bias value
        baised_regular_input = numpy.c_[regular_input, 1] # add a 1 to the input for the bias value
        low_dim_output = baised_sparsed_input * self.sparse_neurons.T
        regular_output = baised_regular_input * self.regular_neurons.T

        self.layer_outputs = self.numpySigmoid(numpy.c_[low_dim_output, regular_output])
        return self.layer_outputs

    def backProp(self, next_layer_deltas, next_layer_neurons, sparse_sample, regular_sample, learning_rate):
        self.layer_deltas = []

        layer_gradient = numpy.multiply(
            self.numpySigDeriv(self.layer_outputs.T), 
            (
                next_layer_deltas.T * next_layer_neurons #[:,0:next_layer_neurons.shape[0]]
            ).T
        )
        sparse_delta_w = layer_gradient[:self.target_dimension,] * learning_rate *  numpy.c_[sparse_sample, 1]
        regular_delta_w = layer_gradient[self.target_dimension:,] * learning_rate *  numpy.c_[regular_sample, 1]

        self.sparse_neurons = self.sparse_neurons + sparse_delta_w
        self.regular_neurons = self.regular_neurons + regular_delta_w


    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def numpySigDeriv(self, x):
        sigdevfunc = numpy.vectorize(self.sigmoidDerivative)
        return sigdevfunc(x)

        return x * (1 - x)

    def numpySigmoid(self, x):
        sigfunc = numpy.vectorize(self.sigmoid)
        return sigfunc(x)

    def sigmoid(self, x):
        # result = 1.0 / ( 1.0 + math.exp(-x/rho) );
        return 1.0 / ( 1.0 + math.exp(-x) )