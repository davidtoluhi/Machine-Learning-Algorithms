classdef vmlp < handle %NB: the "< handle" part indicates inheritance
    properties
        error_rate;
        neurons;
        layer_count;
        layer_outputs;
        layer_deltas;
        layer_neuron_count;
        predicted_labels;%predicted labels from m.test    
        raw_labels;
        errors;
        rh;
    end
        methods
            function train(this, data, labels, learning_rate, iterations, hidden_layer_nodes, rho)
                this.errors = zeros(iterations, 1);
                this.rh=rho;
                this.predicted_labels=zeros(size(labels));
                this.raw_labels=zeros(size(labels));
                input_layer = size(data,2);

                if(sum(hidden_layer_nodes)>0)
                    this.layer_count = size(hidden_layer_nodes, 2)+1; % none for input layer and one for output layer
                else
                    this.layer_count = 2;
                end

                this.neurons = cell(this.layer_count,1);

                if(sum(hidden_layer_nodes)>0)
                    this.layer_neuron_count = [input_layer hidden_layer_nodes 1]; %every number in this vector describes the number of neurons in each layer including the input layer 
                else
                    this.layer_neuron_count = [input_layer input_layer 1]; %every number in this vector describes the number of neurons in each layer including the input layer 
                end
                this.layer_neuron_count
                this.layer_outputs = cell(this.layer_count+1,1); %% +1 because it includes the input layer 
                this.layer_deltas = cell(this.layer_count,1);  %% because no need for input layer
                
                rand('state', sum(100*clock))
                

                for i=1:this.layer_count
                    this.neurons{i}=0.8+2.*randn( this.layer_neuron_count(i+1), this.layer_neuron_count(i)+1);
                    hist(this.neurons{i}, 2)
                end
                
                
                for k=1:iterations
%                     k
                    for j=1:size(data, 1) 
                        current_input=data(j,:);
                        this.layer_outputs{1}=current_input;
                        
                        for i=1:this.layer_count
                               if(i==this.layer_count)
                                   this.layer_outputs{i+1} =[this.layer_outputs{i}, 1]*this.neurons{i}';  %this.neurons(counter_two).identityActivate([this.layer_outputs{i}, 1]);
                               else
                                   this.layer_outputs{i+1} = sigmoid([this.layer_outputs{i}, 1]*this.neurons{i}', rho);%this.neurons(counter_two).activate([this.layer_outputs{i}, 1]);
                               end
                        end
                        
                        net_activation=this.layer_outputs{this.layer_count+1};% +1 because it includes the input layer 
%                         deltas(layer_count)=network_output(j)*(1-network_output(j))*(labels(j)-network_output(j)); %output delta
                        training_err=labels(j)-net_activation;
                        output_delta=training_err;
                        this.layer_deltas{this.layer_count}(1) = output_delta;
                        output_delta_w=learning_rate*output_delta*[this.layer_outputs{this.layer_count} 1];%output delta
                        
                        for p=this.layer_count:-1:2 % going from second-to last layer to first hidden layer
                            this.layer_deltas{p-1} = sigmoidDerivative(this.layer_outputs{p})'.*(this.layer_deltas{p}'*this.neurons{p}(:,1:this.layer_neuron_count(p)))';
                            delta_w=learning_rate*this.layer_deltas{p-1}*[this.layer_outputs{p-1}, 1];
                            this.neurons{p-1} = this.neurons{p-1}+delta_w;
                        end
                        this.neurons{this.layer_count} = this.neurons{this.layer_count}+ output_delta_w;
                        
                    end
                 end
               
            end % end of the train function 
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            function result=predict(this, pattern)
               counter=1;
               this.layer_outputs{1}=pattern;
               
%                for i=1:this.layer_count
%                    if(i==this.layer_count)
%                       for x=1:this.layer_neuron_count(i+1) % bias
%                           this.layer_outputs{i+1}(x)=this.neurons(counter).identityActivate([this.layer_outputs{i}, 1]);
%                           counter=counter+1;
%                       end
%                    else
%                       for x=1:this.layer_neuron_count(i+1) % bias
%                           this.layer_outputs{i+1}(x)=this.neurons(counter).activate([this.layer_outputs{i}, 1]);
%                           counter=counter+1;
%                       end
%                    end
%                end
               for i=1:this.layer_count
                   if(i==this.layer_count)
                      this.layer_outputs{i+1} =[this.layer_outputs{i}, 1]*this.neurons{i}';  %this.neurons(counter_two).identityActivate([this.layer_outputs{i}, 1]);
                   else
                      this.layer_outputs{i+1} = sigmoid([this.layer_outputs{i}, 1]*this.neurons{i}', this.rh);%this.neurons(counter_two).activate([this.layer_outputs{i}, 1]);
                   end
               end
                        
               result=this.layer_outputs{this.layer_count+1};
               
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            function result=test(this, data, labels)
                error=0;
                for i=1:size(data, 1)
                   this.raw_labels(i)=this.predict(data(i,:));
                   error=error+(labels(i)-this.raw_labels(i)).^2;
                end
                result=sqrt(error/size(data,1));
                this.error_rate=result;
            end
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end

end