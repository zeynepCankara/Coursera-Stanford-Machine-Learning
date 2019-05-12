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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% -------------------   Part 1: Cost & Regularization --------------
% For example Theta1.size: 25x400, Theta2.size: 10x26

% Add bias column to X
% a1 = X
X = [ones(m, 1) X]; % new X size: 5000x401

% Obtain activations
z2 = X * Theta1'; % z2 dimension: 5000x25
a2 = sigmoid(z2); % a2 dimension: 5000x25

% add bias unit (+1) unit 0 of layer 2
a2 = [ones(m, 1), a2]; % a2 dimension: 5000x26

z3 = a2 * Theta2'; % z3 dimension: 5000x10
% activation of the 3rd layer a3 = h
h = sigmoid(z3); % h dimension: 5000x10

% recode the y values intp 1xk vectors
yNew = zeros(m, num_labels); % 5000x10
for i = 1:m,
    yNew(i, y(i)) = 1;
end

% Cost function
J = (1/m) * sum ( sum ( (-yNew) .* log(h) - (1-yNew) .* log(1-h) ));

% Regularization term
reg = (lambda/(2*m)) * (sum( sum(Theta1(:, 2:end).^2) ) + sum(sum(Theta2(:, 2:end).^2)) );
J = J + reg;
% ------------------- Part 2: Forward & Backward Propagation ---------------------

% iterate over all examples of the training set
for t = 1:m,
    % add bias unit (col of 1) after calculating activation of a new layer
    a1 = X(t, :)'; % size(a1): 401x1
    z2 = Theta1 * a1; % size(z2): 25x1
          
    a2 = [1; sigmoid(z2)]; % size(a2): 26x1
    z3 = Theta2 * a2; % size(z3): 10x1
    a3 = sigmoid(z3);
    
    % One-hot representation of y values exe: 5 == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] (num classes = 10)
    yNew = (([1:num_labels])== y(t))';  % size(z3): 10x1
    % error between prediction and actual value of y
    delta_3 = a3 - yNew; % size(delta_3): 10x1
    
    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)]; % size(delta_2): 26x1
    delta_2 = delta_2(2:end); % Exclude the bias row
    
    % Inputs not have any error associated with them
    
    % Big delta calculations
    Theta1_grad = Theta1_grad + delta_2 * a1';
    Theta2_grad = Theta2_grad + delta_3 * a2';
end

               
% gradients of the neural network cost function exclude bias column
gradTheta1 = (lambda/m) * Theta1(:, 2:end);
gradTheta2 = (lambda/m) * Theta2(:, 2:end);
               
Theta2_grad = (1/m) * Theta2_grad; % size(10*26)
Theta1_grad = (1/m) * Theta1_grad; % size(25*401)

               
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + gradTheta1;

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + gradTheta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
