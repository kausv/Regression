function [error, error_grad] = regressionObjVal(w, X, y, lambda)

% compute squared error (scalar) and gradient of squared error with respect
% to w (vector) for the given data X and y and the regularization parameter
% lambda