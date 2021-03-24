function [c_idx, center, cost] = mykmeans(X, k, no_iter, initializer)
% X: input features
% k: number of clusters
% no_iter: number of iterations
% init:   1: naive
%         2: k-means ++
switch nargin
    case 1
        k = 2;
        no_iter = 10;
        initializer = 1;
    case 2
        no_iter = 10;
        initializer = 1;
    case 3
        initializer = 1;
end
[x_length, y_length] = size(X);
c_idx = zeros(y_length, 1);
% initialization
center = k_means_init(X, k, initializer);
for i = 1:no_iter
    % the k-means clustering algorithm
    %% Assign each data point to its closest cluster center: E-step
    for j = 1: x_length
        % computet the cost of each point and center
        cost = sum((X(j, :) - center) .^ 2, 2);
        % assign it to the nearest center
        [~, idx] = min(cost);
        c_idx(j) = idx;
    end
 
    %% Update each cluster center by computing the mean of all points assigned to it: M-step
    for m = 1:k
        in_same_cluster = X(c_idx == m, :);
        center(m, :) = mean(in_same_cluster);
    end 
end

% calculate the cost
cost = 0;
for i = 1:k
    cost = cost + sum(norm(X(c_idx==i,:)-center(i,:)).^2);
end
end % end of k-means

