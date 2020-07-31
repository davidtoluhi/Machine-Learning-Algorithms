from vmlp_multiclass import vmlp as multiclass_neural_network
from embedded_layer import EmbeddedLayer
import numpy

data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]]) # input data
labels = numpy.matrix([[0,0,0],[1,0,0],[1,0,0],[0,0,0]]) # labels

user_model = multiclass_neural_network(data, labels, [2], 0.1, 2000) 
user_model.train() 
# multiclass_neural_network.rawLabels()
user_model.predictedLabels()
