function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);
unreg_cost = 1./m * ( -y' * log(h) - ( 1 - y' ) * log ( 1 - h) );
theta(1) = 0;
theta_sqr = theta' * theta;
reg_cost = theta_sqr * (lambda / (2 * m));
J = unreg_cost + reg_cost;

unreg_grad = 1./m * X' * (h - y);
reg_grad = theta * (lambda / m);

grad = unreg_grad + reg_grad;


% =============================================================

end
