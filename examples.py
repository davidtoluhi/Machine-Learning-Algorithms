from vmlp_multiclass import vmlp as multiclass_neural_network
from embedded_layer import EmbeddedLayer
import numpy

data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]]) # input data
labels = numpy.matrix([[0,0,1],[1,0,0],[1,0,0],[0,0,1]]) # labels

user_model = multiclass_neural_network(data, labels, [2], 0.1, 2000) 
# user_model.train() 
# multiclass_neural_network.rawLabels()
# user_model.predictedLabels()


sparse_data = numpy.matrix([
    [0, 0, 0, 0, 12, 108],
    [0, 0, 0, 1, 12, 108],
    [0, 0, 1, 0, 12, 108],
    [0, 0, 1, 1, 12, 108],
    [0, 1, 0, 0, 12, 108],
    [0, 1, 0, 1, 12, 108],
    [0, 1, 1, 0, 12, 108],
    [0, 1, 1, 1, 12, 108],
    [1, 0, 0, 0, 12, 108],
    [1, 0, 0, 1, 12, 108],
    [1, 0, 1, 0, 12, 108],
    [1, 0, 1, 1, 12, 108],
    [1, 1, 0, 0, 12, 108],
    [1, 1, 0, 1, 12, 108],
    [1, 1, 1, 0, 12, 108],
    [1, 1, 1, 1, 12, 108],
])
sparse_labels = numpy.matrix([
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
])
embedded_model = multiclass_neural_network(sparse_data, sparse_labels, [2], 0.1, 2000)
embedded_layer = EmbeddedLayer(4, 2, 2, 3)
embedded_model.embedLayer(embedded_layer)
embedded_model.train() 
embedded_model.predictedLabels()





def hamm(length):
    hammed_dist = 2 ** length 
    rays = numpy.zeros([hammed_dist, length])
    for i in range(0, hammed_dist):
        bin_vector = list(bin(i)[2:])
        if len(bin_vector) < length: 
            vector = [0] * (length - len(bin_vector))
            rays[i] = vector + bin_vector
        else: 
            rays[i] = bin_vector

    for i in range(0,len(rays)):
        print(rays[i])
        
# hamm(4)