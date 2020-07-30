# Machine-Learning-Algorithms
A repository for a personal implementation of machine learning algorithms........ I re-invent the wheel every now and then because "What I cannot create, I cannot understand." - Richard Feynman

The main item currently in this repository is the vmlp.py file: A python version of a neural network I created in matlab, what's great about it is that you can just create a a neural net by providing a vector consisting of the amount of neurons in each layer(except the output) of the neural net 

# Requirements 
python 2.7 and numpy 

# Example Usage for one hot encoding
## Constructor parameters: 
```vmlp (<numpy matrix of input data>, <numpy matrix of corresponding labels>, <vector of hidden layer neurons>, <learning_rate>, <number of iterations>)```

```
...
from vmlp import vmlp
...
data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]]) # xor dataset input data
labels = numpy.matrix([[0],[1],[1],[0]]) # xor dataset labels

user_model = vmlp(data, labels, [2], 0.1, 4000) # alternative example: user_model = vmlp(data, labels, [2, 5, 69, 2], 0.1, 4000) this is to illustrate the usage of the hidden layer neuron count parameter
user_model.train() # this trains the model 
user_model.faTest(data, labels) # performs function approximation test on the provided data
user_model.patregTest(data, labels) # performs a pattern recognition test on the provided data
```
Access the predicted labels through ```user_model.predictedLabels()``` and the error rate through ```user_model.error_rate``` after testing.


# Example Usage for multiclass labelling(using softmax - there is only one "right answer" = the outputs are mutually exclusive)
## Constructor parameters: 
```vmlp (<numpy matrix of input data>, <numpy matrix of corresponding labels>, <vector of hidden layer neurons>, <learning_rate>, <number of iterations>, <use_softmax=True>,  <use_numbers=False>, <logit_layer_node_count=0>)```
```
...
from vmlp_multiclass import vmlp
...
data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]]) # input data
labels = numpy.matrix([[1,0,0],[0,1,0],[0,0,1],[0,0,1]]) # labels

user_model = vmlp(data, labels, [2], 0.1, 4000) # alternative example: user_model = vmlp(data, labels, [2, 5, 69, 2], 0.1, 4000) this is to illustrate the usage of the hidden layer neuron count parameter

alertantively (using number positions for the labels): 
data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]]) # input data
labels = numpy.matrix([[0],[1],[2],[2]) # labels
user_model = vmlp(data, labels, [2, 3], 0.1, 4000, True, True, 3) 


user_model.train() # this trains the model 
user_model.patregTest(data, labels) # performs a pattern recognition test on the provided data
```
Access the predicted labels through ```user_model.predictedLabels()``` and the error rate through ```user_model.error_rate``` after testing.




# Example Usage for multiclass labelling(using sigmoid - there is more than one "right answer" = the outputs are NOT mutually exclusive)
## Constructor parameters: 
```vmlp (<numpy matrix of input data>, <numpy matrix of corresponding labels>, <vector of hidden layer neurons>, <learning_rate>, <number of iterations>, <use_softmax: defaults to true>, <use_numbers=False>, <logit_layer_node_count=0>)```
```
...
from vmlp_multiclass import vmlp
...
data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]]) # input data
labels = numpy.matrix([[1,0,0],[0,1,0],[0,0,1],[0,0,1]]) # labels

user_model = vmlp(data, labels, [2], 0.1, 4000, False) 


alertantively (using number positions for the labels): 
data = numpy.matrix([[0,0],[0,1],[1,0],[1,1]]) # input data
labels = numpy.matrix([[0],[1],[2],[2]) # labels
user_model = vmlp(data, labels, [2, 3], 0.1, 4000, False, True, 3) 


user_model.train() # this trains the model 
user_model.patregTest(data, labels) # performs a pattern recognition test on the provided data
```
Access the predicted labels through ```user_model.predictedLabels()``` and the error rate through ```user_model.error_rate``` after testing.
