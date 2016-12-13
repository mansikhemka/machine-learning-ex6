function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

results = eye(64,3);
errorRow = 0;






result =[];
c=[0.01 0.03 0.1 0.3 1 3 10 30];
sig=[0.01 0.03 0.1 0.3 1 3 10 30];

for i=1: size(c,2)
	for j=1: size(sig,2)
		model = svmTrain(X, y, c(i), @(x1, x2) gaussianKernel(x1, x2, sig(j))); 
	    predictions = svmPredict(model, Xval);
	    result =  [result; c(i) sig(j) mean(double(predictions ~= yval))];
		end
	end


[l,m] = min(result, [] ,1);

k=result(m(3),:);
C = k(1);
sigma = k(2);










% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
