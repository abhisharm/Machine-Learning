function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

%Computing Cost
diff = ((X*theta)-y); 
cost = (1/(2*m)).*(diff.^2);
J = J + sum(cost,1);
theta_cost = ((lambda/(2*m)).*(theta.^2));
theta_cost(1,:) = 0; 
J = J + sum(theta_cost); 

%Computing the gradient
grad = grad + (1/m).*(X'*diff); 
theta_grad = (lambda/m)*theta; 
theta_grad(1,:) = 0; 
grad = grad + theta_grad; 










% =========================================================================

grad = grad(:);

end
