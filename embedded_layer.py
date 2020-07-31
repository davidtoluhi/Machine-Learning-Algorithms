import numpy 
import math

class EmbeddedLayer(object): 


    def __init__(self, sparse_input, target_low_dimension, total_neurons_in_layer): 

        self.target_dimension = target_low_dimension
        self.target_neurons = []
        self.total_neurons_in_layer = total_neurons_in_layer
        self.layer_outputs = []
        self.target_neurons = numpy.matrix( 
                numpy.random.random(
                    (self.target_low_dimension, # number of neurons in this layer
                    self.total_neurons_in_layer+1) # weights for the neurons in this layer: number of neurons in previous layer plus a bias 
                )
            )
        
    def feedForward(self, sparse_input, regular_input):
        baised_regular_input = numpy.c_[regular_input, 1] # add a 1 to the input for the bias value

        low_dim_output = sparse_input * self.neurons[:self.target_dimension]
        regular_output = regular_input * self.neurons[self.target_dimension:]

        self.layer_outputs = numpySigmoid(numpy.c_[low, regular_output])
        return self.layer_outputs

    def backProp(self, next_layer_deltas, next_layer_neurons, sparse_sample, regular_sample):
        self.layer_deltas = []
        self.layer_deltas.append(output_delta)

        layer_gradient = numpy.multiply(
            self.numpySigDeriv(self.layer_outputs), 
            (
                next_layer_deltas * self.next_layer_neurons[:,0:next_layer_neurons.shape[0]]
            ).T
        )

        sparse_delta_w = layer_gradient * self.learning_rate *  sparse_sample
        regular_delta_w = layer_gradient * self.learning_rate *  numpy.c_[regular_sample, 1]

        self.neurons[:self.target_low_dimension] = self.neurons[:self.target_low_dimension] + sparse_delta_w
        self.neurons[self.target_low_dimension:] = self.neurons[self.target_low_dimension:] + regular_delta_w


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