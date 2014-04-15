function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1
%(ID + X>X)?1X>y
D = size(X,2);
w = (lambda * eye(D) + X'*X)\(X'*y);


end