function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Goal: make sure theta0 does not affect by regularization term
% not regulariza theta0
avoidReg = theta;
% set theta0 to 0 in order to avoid regularization
avoidReg(1) = 0;

% Calculate the cost
J = (-1 / m) * sum(y .* log(sigmoid(X * theta)) + (1 - y) .* log(1 - sigmoid(X * theta))) + (lambda / (2 * m)) * sum(avoidReg.^2);

hyphothesis = sigmoid(X * theta);
% Calculate the gradient
grad = (1/m) .* (X' * (hyphothesis - y)) + (lambda / m) * avoidReg;

% =============================================================

end
