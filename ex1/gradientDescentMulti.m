function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    tmp = X * theta;
    tmp = y - tmp;
    tmp_theta = zeros(length(tmp),1); 
    for k = 1:size(X, 2)
        tmp_theta(:,k)=tmp; 
    end
    tmp_theta = tmp_theta .* X; 
    tmp_theta = sum(tmp_theta, 1)'; 
    theta = theta + tmp_theta.*(alpha/m); 
    J_history(iter) = computeCostMulti(X, y, theta);
end

end