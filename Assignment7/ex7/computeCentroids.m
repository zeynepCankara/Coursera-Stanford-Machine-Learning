function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%


% keep position of elements have centroid k
sum_position_k = zeros(K, n);
% count number of elements have centroid k
count_centroid_k = zeros(K, 1);
for k = 1:size(idx, 1)
    % get the cluster number k
    z = idx(k);
    % increment the number of the cluster k
    count_centroid_k(z, 1) += 1;
    % increment position of the cluster k
    sum_position_k(z, :) += X(k, :);
end

% calculate mean of the clusters
centroids = sum_position_k ./ count_centroid_k;

% =============================================================


end

