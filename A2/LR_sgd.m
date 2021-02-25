

% random initial point:w and b
w=rand(size(X));
b= rand(1);
learning_rate = 0.001;
no_iterations = 1000
for i=1:length(X)
    LR_re_predict(w, b, X(i,:), Y(i),learning_rate, no_iterations)
end

%% --------------------------------------------------

%% Model optimization
function output = LR_re_opt(w, b, X, Y,learning_rate, no_iterations)
    for i =1:no_iterations
        % predicted result
        Y_hat = my_sigmod(dot(w,X)+b);
        cost = loss_cross_entropy(Y_hat,Y);  % + REGULATION_TERM

        % Gradient calculation
        dw = (Y-Y_hat).*X;
        db = (Y-Y_hat).X;

        %% update parameter 
        w_new = w - learning_rate.*dw;
        db = b - learning_rate.*db;
    end    
end

%% Predict model
function [w_hat, b_hat] = LR_re_predict(w, b, X, Y, learning_rate, no_iterations)
    
    
end


%% ---------------------------------------------------------------
function loss = loss_cross_entropy(y_estimate, y_actual) 
    loss = -y_actual*log(y_estimate)-(1-y_actual)*log(1-y_estimate);
end

function prob = my_sigmod(z)
    prob = 1/(1+exp(-z));
end

function [J, grad] = cost(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = my_sigmod(X*theta);
theta_shift = [0; theta(2:end, :)];

J = ((1/m) * ((-y' * log(h)) - ((1-y)' * log(1-h)))) + (lambda/(2*m) * (theta_shift' * theta_shift));
z
grad = ((1/m) * X' * (h-y)) + ((lambda/m) * theta_shift);
grad = grad(:);

end