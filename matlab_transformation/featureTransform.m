function [YAndOutlier, param] = featureTransform(YAndOutlier, XAndOutlier, ptsIdxRefined)
% featureTransform returns the transformed moving point set YAndOutlier,
%       and the transformation parameter param. It warps the whole 
%       moving points YAndOutlier by the regularized 
%       Gaussian-radial-basis-function based energy function to ensure 
%       that the preserved inliers are aligned while the outliers are pulled 
%       simultaneously.
%
%%=====================================================================
%% $Author: M.E. Su ZHANG, supervised by Assoc. Yang YANG. $
%% $Date: Mon, 31 Dec 2018$
%% $Contact: sorazcn@gmail.com$
%%=====================================================================
%     beta = config.beta; 
%     lambda = config.lambda;
    config = initilizationSIR;
    viz = config.viz;
%     kernel = computeKernelAndBasis(Y, beta);
%     basis = computeKernelAndBasis(YAndOutlier, Y, beta);
    X=XAndOutlier(ptsIdxRefined,:); 
    inlierNum = size(X, 1); ptsNum = size(XAndOutlier, 1);
    for i = 1:1
        Y=YAndOutlier(ptsIdxRefined,:);
        if viz
            clf;
            figure(1);
            plot(YAndOutlier(:,1),YAndOutlier(:,2),'.','LineWidth',2,'color','k','MarkerSize',5);hold on;
            plot(XAndOutlier(:,1),XAndOutlier(:,2),'.','LineWidth',2,'color','r','MarkerSize',6);hold on;
            plot(Y(:,1),Y(:,2),'o','LineWidth',1,'color','k','MarkerSize',5);hold on;
%             plot(X(:,1),X(:,2),'o','LineWidth',2,'color','r','MarkerSize',5.5);hold on;
            pause(1);
        end
%         param = pinv(kernel + lambda * eye(inlierNum)) * (X-Y);
%         YAndOutlier = YAndOutlier + basis * param; 
%         YAndOutlier=approxThinPlateSpline(Y,X, YAndOutlier, XAndOutlier, ptsIdxRefined);
        [YAndOutlier, param] = thinPlateSpline(Y,X, YAndOutlier, config);
    end  