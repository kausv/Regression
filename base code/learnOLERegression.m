function w = learnOLERegression(X,y)

% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1
D = size(X,2);

w = (X'*X)\(X'*y);

% epsilon = sqrt(6) / sqrt(D + 1);
% w = rand(1, D) * 2* epsilon - epsilon;
% 
% Wt = X * w;
% delta = y - Wt;


end