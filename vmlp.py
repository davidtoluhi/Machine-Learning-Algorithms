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
    layer_deltas = []
    weight_updates = []
    predicted_labels = []
    raw_labels = []
    error_rate = 1

    """docstring forvmlp."""
    def __init__(self, data, labels, hidden_layer_nodes_list_rep, learning_rate, iterations, weight_range):
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
            # numpy.random.random(rows, cols)
            # initializing weights of vectors/neurons
            layer_neurons = numpy.matrix( numpy.random.random((self.layer_neuron_count[i+1], self.layer_neuron_count[i]+1)))
            self.neurons.append(layer_neurons)
            self.weight_updates.append(layer_neurons)
        # shape is a property of both numpy ndarray's and matrices.
        # A.shape
        # will return a tuple (m, n), where m is the number of rows, and n is the number of columns

    def feedForward(self, input_):
        self.layer_outputs = []
        self.layer_outputs.append(input_)
        for z in range(0, self.layer_count):
            # if z == self.layer_count-1:
            #     self.layer_outputs[i+1] = self.layer_outputs[z] * self.neurons[z].T# i+1 because we want the input layer untouched
            # else:

            # v = np.ones((10, 1))
            # c = np.c_[m, v]

            inp = numpy.c_[self.layer_outputs[z], 1]
            self.layer_outputs.append(self.numpySigmoid(inp * self.neurons[z].T))
        pass

    def backpropInput(self, label):
        net_activation = self.layer_outputs[self.layer_count][0,0] # because it includes the input layer
        training_err = label - net_activation
        output_delta = training_err
        self.layer_deltas = []
        self.layer_deltas.append(output_delta)
        output_delta_w = self.learning_rate * output_delta * numpy.c_[self.layer_outputs[self.layer_count-1], 1]
        # output_delta_w = self.learning_rate * output_delta * self.layer_outputs[self.layer_count-1]
        # print (output_delta_w)
        # print(self.neurons)
        for i in range(self.layer_count, 1, -1):
            # print(self.layer_neuron_count[i])
            # print(self.layer_outputs[i-1])
            # print '--'
            # print(self.neurons[i-1])
            # print((self.neurons[i-1][:,0:self.layer_neuron_count[i-1]]).T)
            # print((self.layer_deltas[0][0,0] * self.neurons[i-1][:,0:self.layer_neuron_count[i-1]]).T)

            self.layer_deltas.insert(0, numpy.multiply(self.numpySigDeriv(self.layer_outputs[i-1].T) , (self.layer_deltas[0].T * self.neurons[i-1][:,0:self.layer_neuron_count[i-1]]).T)) # bias

            # print(self.layer_deltas[0])

            delta_w = self.learning_rate * self.layer_deltas[0] * numpy.c_[self.layer_outputs[i-2], 1]
            # print(delta_w)
            # print '**'
            # print(self.neurons[i-1])
            # self.weight_updates[i-1] = delta_w
            self.neurons[i-2] =  delta_w + self.neurons[i-2]

        # for i in range(self.layer_count, 1, -1):
        #     self.neurons[i-1] = weight_updates[i-1] + self.neurons[i-1]
        # print(self.neurons[self.layer_count-1])

        self.neurons[self.layer_count-1] = self.neurons[self.layer_count-1] + output_delta_w

        pass

    def train(self):

        for x in range(0, self.iterations):
            for y in range(0, self.data.shape[0]):
                # v = np.ones((10, 1))
                # c = np.c_[m, v]

                input_ = self.data[y, :]
                self.feedForward(input_)
                self.backpropInput(self.labels[y])
        pass

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
        # for y in range(0, data.shape[0]):
            # input_ = self.data[y, :]
            # self.feedForward(input_)
            # print(self.layer_outputs[self.layer_count][0,0])
        self.patregTest(self.data, self.labels)
        print(self.raw_labels)
        pass

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
        self.error_rate=result;
        return result;

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
#
# # problem is the difference in sigmoid prediction
# neural_net = vmlp(data, labels, [2], 0.1, 4000)
# neural_net.train()
# neural_net.predictedLabels()
