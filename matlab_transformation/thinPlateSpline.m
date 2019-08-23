function [YAndOutlier, param] = thinPlateSpline(Y, X, YAndOutlier, config)
%thinPlateSpline performs the TPS transformation. 

%%=====================================================================
%% $Author: M.E. Su ZHANG, supervised by Assoc. Yang YANG. $
%% $Date: Mon, 31 Dec 2018$
%% $Contact: sorazcn@gmail.com$
%%=====================================================================

inlierNum = size(Y,1);
K = repmat(Y, [1 1 inlierNum]) - repmat(permute(Y, [3 2 1]), [inlierNum 1 1]);
K = max(squeeze(sum(K.^2, 2)), eps);
lambda = config.lambda;
if size(Y, 2) == 2
    K = K.* log(sqrt(K));  
    P = [ones(inlierNum,1), Y]; 
    L = [ [K+lambda*eye(inlierNum), P];[P', zeros(3,3)] ]; 
    param = pinv(L) * [X; zeros(3,2)]; 

    inlierAndOutlierNum=size(YAndOutlier,1); 
    K = repmat(YAndOutlier, [1 1 inlierNum]) - repmat(permute(Y, [3 2 1]), [inlierAndOutlierNum 1 1]);
    K = max(squeeze(sum(K.^2, 2)), eps);

    K = K.* log(sqrt(K));
    P = [ones(inlierAndOutlierNum,1), YAndOutlier];
    L = [K, P];
    YAndOutlier = L * param;    
else

    K = sqrt(K); 
    P = [ones(inlierNum,1), Y]; 
    L = [ [K+lambda*eye(inlierNum), P];[P', zeros(4,4)] ]; 
    param = pinv(L) * [X; zeros(4,3)]; 

    inlierAndOutlierNum=size(YAndOutlier,1); 
    K = repmat(YAndOutlier, [1 1 inlierNum]) - repmat(permute(Y, [3 2 1]), [inlierAndOutlierNum 1 1]);
    K = max(squeeze(sum(K.^2, 2)), eps);

    K = sqrt(K); 
    P = [ones(inlierAndOutlierNum,1), YAndOutlier];
    L = [K, P];
    YAndOutlier = L * param;
end