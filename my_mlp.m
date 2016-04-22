classdef my_mlp < handle %NB: the "< handle" part indicates inheritance
    properties
        error_rate;
        neurons;
        layer_count;
        layer_outputs;
        layer_neuron_count;
        predicted_labels;%predicted labels from m.test    
        raw_labels;
    end
        methods
           
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % /*=====================================
            %  ||         Hyper Parameters          ||
            %  |  --             --            --    |
            %  |           i. Layer count            |
            %  |          ii. Neuron Count           |
            %  |         iii. Learning rate          |
            %  |           iv. Iterations            |
            %  |  --                           --    |
            %  ======================================*/
            
            
            function train(this, data, labels, learning_rate, iterations, hidden_layer_nodes)
                
%               your neurons can be an array of weights with 
%  %            each row representing the weights of one neuron, but 
%  %            remember the breaking into tasks thing r.neville was telling you about
%               layer_nodes format: [3 4 4 5] each element in the array;
%  %            strictly for the hidden layers
%               output layer: soley determined by my model: regression 
%  %            (one node) versus classification(number of nodes equivalent
%  %            to the number of classes, assuming softmax)
%               if nargin< 5
%                  iterations=100*size(data,1);
%               end

% http://www.cse.unsw.edu.au/~cs9417ml/MLP2/
            
                this.predicted_labels=zeros(size(labels));
                this.raw_labels=zeros(size(labels));
                first_layer = size(data,2);

                if(sum(hidden_layer_nodes)>0)
                    this.layer_count = size(hidden_layer_nodes, 2)+2;
                else
                    this.layer_count = 2;
                end
               
                network_neuron_count = first_layer+sum(hidden_layer_nodes)+1; %+1; to account for the output neuron... since this is a regression network, it is only 

                network_neurons(network_neuron_count) = neuron(first_layer+1);

                if(sum(hidden_layer_nodes)>0)
                    this.layer_neuron_count = [first_layer first_layer hidden_layer_nodes 1]; %every number in this vector describes the number of neurons in each layer including the input layer 
                else
                    this.layer_neuron_count = [first_layer first_layer 1]; %every number in this vector describes the number of neurons in each layer including the input layer 
                end
                this.layer_outputs = cell(this.layer_count+1,1); %% +1 because it includes the input layer 

%                 deltas=cell(this.layer_count, 1);
                counter_one=1;
                for i=1:this.layer_count
                    for j=1:this.layer_neuron_count(i+1)  % network_neuron_count
                        network_neurons(counter_one)=neuron(this.layer_neuron_count(i)+1); %'+1' is to account for bias 
                        counter_one=counter_one+1;
                    end
                end
           
                for k=1:iterations
                    for j=1:size(data, 1) 

                        current_input=data(j,:);
                        counter_two=1;
                        this.layer_outputs{1}=current_input;

                        for i=1:this.layer_count
                               
%                                this.layer_outputs{i+1}(1)=1; % bias
                               for x=1:this.layer_neuron_count(i+1) % bias
                                   this.layer_outputs{i+1}(x)=network_neurons(counter_two).activate([this.layer_outputs{i}, 1]);
                                   counter_two=counter_two+1;
                               end
                        end
                        
                        net_activation=this.layer_outputs{this.layer_count+1};% +1 because it includes the input layer 
    %                     deltas(layer_count)=network_output(j)*(1-network_output(j))*(labels(j)-network_output(j)); %output delta
                        training_err=labels(j)-net_activation;
                        output_delta=sigmoidDerivative(net_activation)*training_err;
                        output_delta_w=learning_rate*output_delta*[this.layer_outputs{this.layer_count} 1];%output delta
                        
                        
                        counter_three=network_neuron_count-1;
% there's something going on with your learning rate
                        for p=this.layer_count:-1:2
                            for jj=this.layer_neuron_count(p):-1:1  % network_neuron_count
                                network_neurons(counter_three).backProp(network_neurons(network_neuron_count).weights(jj), output_delta, learning_rate);
                                counter_three=counter_three-1;
                            end
                        end
                        network_neurons(network_neuron_count).weights=network_neurons(network_neuron_count).weights + output_delta_w;
                        
                    end
                end
                this.neurons=network_neurons;
                this.error_rate=this.test(data, labels);
            end % end of the train function 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            function result=predict(this, pattern)
               counter=1;
               this.layer_outputs{1}=pattern;
               
               for i=1:this.layer_count
                   for x=1:this.layer_neuron_count(i+1)
                       this.layer_outputs{i+1}(x)=this.neurons(counter).activate([this.layer_outputs{i} 1]);
                       counter=counter+1;
                   end
               end
         result=this.layer_outputs{this.layer_count+1};
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            function result=test(this, data, labels)
                e=0;
                for i=1:size(data, 1)
                   this.raw_labels(i)=this.predict(data(i,:));
                   activation=0;
                   
                   if(this.raw_labels(i)>0.5)
                       activation=1;
                   end
                   this.predicted_labels(i)=activation;
                   if(activation~=labels(i))
                       e=e+1;
                   end
                end
                result=e/size(data,1);
            end
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end

end