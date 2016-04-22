classdef neuron < handle
    
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        weights;
        activation;
        input;
    end
    
    methods
        function obj=neuron(weight_count) % weight_count bias_inclusive
            if nargin~=0
                rand('state', sum(100*clock))
                obj.weights=randn(1, weight_count);
            end
        end
        
        function result=activate(this, input)
            this.input=input;
            this.activation=this.sigmoid(this.weights*input');
%             if(temp>0.5)   ---- only used in testing 
%                 this.activation=1;
%             else 
%                 this.activation=0;
%             end
            result=this.activation;
        end
        
%         multiply deltas from forward hidden layer 
        function backProp(this, output_weights, output_deltas,learning_rate)  
           delta=sigmoidDerivative(this.activation)*sum(output_deltas.*output_weights);
           delta_w=learning_rate*delta*this.input;
           this.weights=this.weights+delta_w;
        end
        
        function result=sigmoid(this, activation)
            result=1/(1+exp(-(activation)));
        end
    end
    
end

