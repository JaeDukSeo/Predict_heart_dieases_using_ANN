function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

act_one = [ones(m,1) X];
z_two = act_one * Theta1';
act_two = sigmoid(z_two);
act_two = [ones(m,1) act_two];
z_three = act_two * Theta2';
act_three = sigmoid(z_three);
h_theta = act_three;

for k = 1:num_labels
    ytemp = (y==(k-1));
    J = J + sum((-ytemp' * log(h_theta(:,k)) - (1 - ytemp)' * log(1 - h_theta(:,k) ) )) ;
end
% Adding the Regularized Gradient
J = J/m;
reg = 0;
reg2 = 0;
for j=1:size(Theta1,1)
    for k = 2:size(Theta1,2)
        reg = reg + Theta1(j,k)^2;
    end
end
for j=1:size(Theta2,1)
    for k = 2:size(Theta2,2)
        reg2 = reg2 + Theta2(j,k)^2;
    end
end
J = J + (reg + reg2) * ( lambda/ (2*m));
% Getting the Gradient Derivative
delta_three = zeros(1,num_labels);
for i  = 1:m
    for k = 1:num_labels
        ytemp = (y(i)==(k-1));
        delta_three(k) = h_theta(i,k) - ytemp;
    end
  
    delta_two = (Theta2(:,2:end)' * delta_three') .*sigmoidGradient(z_two(i,:))';
    
    Theta1_grad = Theta1_grad +delta_two*act_one(i,:);
    Theta2_grad = Theta2_grad + delta_three'*act_two(i,:);
end
% A better way to do the computation
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
