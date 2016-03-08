function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

fprintf('theta: %f\n', theta);
fprintf('number of training examples: %f\n', m);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h = sigmoid(X);
red = sum(-y.* log(h));
fprintf('RED: %f\n', red);


blue = sum((1-y).* log(1 - h));
fprintf('blue: %f\n', blue);

redMinusBlue = red - blue;

fprintf('redMinusBlue: %f\n', redMinusBlue);

uCost = 1/m * redMinusBlue;

fprintf('uCost: %f\n', uCost);

fprintf('theta(1): %f\n', theta(1));
theta(1) = 0;
theta_sqr = theta.^2;
fprintf('theta_sqr: %f\n', theta_sqr);


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


% =============================================================

end
