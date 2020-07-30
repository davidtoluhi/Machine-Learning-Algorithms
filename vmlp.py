import numpy
import math
'''
VMLP: Vectorised Multilayer Perceptron 
    a neuron is represented as a vector; 
    a the neural network is represented as an array of a matrix of vectors
    this helps making the training process faster;
    todo: use MinPy to leverage GPU support 
'''
class vmlp(object):
    # tensorflow or scikit learn?
    # you could also use clustering or k-means
    neurons = []
    layer_count = 2
    input_layer_neuron_count = 0
    layer_neuron_count = []
    data = []
    labels = []
    learning_rate = 0.1
    iterations = 1000
    layer_outputs = []
    layer_gradients = []
    weight_updates = []
    predicted_labels = []
    raw_labels = []
    error_rate = 1

    """docstring forvmlp."""
    def __init__(self, data, labels, hidden_layer_nodes_list_rep, learning_rate, iterations, weight_range=[0,2]):
        super(vmlp, self).__init__()
        # self.arg = arg
        self.input_layer_neuron_count = data.shape[1]
        self.data = data
        self.labels = labels
        self.predicted_labels = numpy.zeros(labels.shape[0])
        self.raw_labels = numpy.zeros(labels.shape[0])
        self.learning_rate = learning_rate
        self.iterations = iterations
        
        if sum(hidden_layer_nodes_list_rep) > 0:
            self.layer_count = len(hidden_layer_nodes_list_rep) + 1 # none for input layer and one for output layer
            self.layer_neuron_count = [self.input_layer_neuron_count] + hidden_layer_nodes_list_rep + [1] #output neuron
        else:
            self.layer_neuron_count = [self.input_layer_neuron_count] + [self.input_layer_neuron_count] +[1]

        for i in range(0, self.layer_count):
            # initializing weights of vectors/neurons
            layer_neurons = numpy.matrix( numpy.random.random((self.layer_neuron_count[i+1], self.layer_neuron_count[i]+1)))
            self.neurons.append(layer_neurons)
            self.weight_updates.append(layer_neurons)

    def set_weight_range(weight_range):
        self.weightRangeIsSet = True 
        self.weight_range = weight_range

    def feedForward(self, input_):
        self.layer_outputs = []
        self.layer_outputs.append(input_)
        for layer in range(0, self.layer_count):
            inp = numpy.c_[self.layer_outputs[layer], 1]
            self.layer_outputs.append(self.numpySigmoid(inp * self.neurons[layer].T))
        

    def backpropInput(self, label, sample):
        net_activation = self.layer_outputs[self.layer_count][0,0] # because it includes the input layer
        training_err = label - net_activation
        output_delta = training_err
        self.layer_gradients = []
        self.layer_gradients.append(output_delta)

        output_delta_w = self.learning_rate * output_delta * numpy.c_[self.layer_outputs[self.layer_count-1], 1]
        for i in range(self.layer_count-1, 0, -1):

            self.layer_gradients.insert(
                0, # input it in position 0 because we're iterating backwards in terms of layers 
                numpy.multiply(
                    self.numpySigDeriv(self.layer_outputs[i].T) , 
                    (
                        self.layer_gradients[0].T * self.neurons[i][:,0:self.layer_neuron_count[i]] 
                    ).T
                )
            ) 

            delta_w = self.learning_rate * self.layer_gradients[0] * numpy.c_[self.layer_outputs[i-1], 1]
            self.neurons[i-1] =  delta_w + self.neurons[i-1]
      
        self.neurons[self.layer_count-1] = self.neurons[self.layer_count-1] + output_delta_w

        

    def train(self):

        for x in range(0, self.iterations):
            for y in range(0, self.data.shape[0]):
                sample = self.data[y, :]
                self.feedForward(sample)
                self.backpropInput(self.labels[y], sample)
        

    def numpySigDeriv(self, x):
        sigdevfunc = numpy.vectorize(self.sigmoidDerivative)
        return sigdevfunc(x)

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def numpySigmoid(self, x):
        sigfunc = numpy.vectorize(self.sigmoid)
        return sigfunc(x)

    def sigmoid(self, x):
        # result = 1.0 / ( 1.0 + math.exp(-x/rho) );
        return 1.0 / ( 1.0 + math.exp(-x) )

    def predictedLabels(self):
        self.patregTest(self.data, self.labels)
        print(self.raw_labels)
        

    def predict(self, input_vector):
        self.feedForward(input_vector)
        return self.layer_outputs[self.layer_count][0,0]

    def faTest(self, data, labels):
        error=0
        # self.raw_labels=zeros(1,size(labels, 1));
        for i in range(0, data.shape[0]):
            self.raw_labels[i] = self.predict(data[i,:])
            error = error + (labels[i] - self.raw_labels[i])**2
        result = math.sqrt(error/data.shape[0])
        self.error_rate=result
        return result

    def patregTest(self, data, labels):
        error=0
        for i in range(0, data.shape[0]):
            self.raw_labels[i]=self.predict(data[i,:])
            activation=0
            if self.raw_labels[i] > 0.5:
                activation=1
            self.predicted_labels[i]=activation
            if activation != labels[i]:
                error = error+1
            self.error_rate = error/data.shape[0]
        return error/data.shape[0]

#
# data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]])
# labels = numpy.matrix([[0],[1],[1],[0]])

# # problem is the difference in sigmoid prediction
# neural_net = vmlp(data, labels, [2,2], 0.1, 7000)
# neural_net.train()
# neural_net.predictedLabels()
