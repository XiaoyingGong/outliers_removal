function [nMovLoose, nFixLoose, normal]=sirNorm2(movLoose, fixLoose, D)
% sirNorm2 normalizes the feature point sets to have zero mean and unit
%       standard variation.

%%=====================================================================
%% $Author: M.E. Su ZHANG, supervised by Assoc. Yang YANG. $
%% $Date: Sat, 30 June 2018$
%% $Contact: sorazcn@gmail.com$
%%=====================================================================
    M = size(movLoose, 1); N = size(fixLoose, 1);

    if D == 2

        normal.movMean=mean(movLoose); normal.fixMean=mean(fixLoose);
        movPts=movLoose-repmat(normal.movMean,M,1);
        fixPts=fixLoose-repmat(normal.fixMean,N,1);

        normal.movScale=sqrt(sum(sum(movPts.^2,2))/M);
        normal.fixScale=sqrt(sum(sum(fixPts.^2,2))/N);

        movPts=movPts/normal.movScale;
        fixPts=fixPts/normal.fixScale;

        nMovLoose=movPts;
        nFixLoose=fixPts;
        
    elseif D == 4
        
        nMovLoose=movLoose;
        nFixLoose=fixLoose;

        movPts = movLoose(:,1:2); 
        fixPts = fixLoose(:,1:2); 

        M = size(movPts, 1); N = size(fixPts, 1);
        normal.movMean=mean(movPts); normal.fixMean=mean(fixPts);

        movPts=movPts-repmat(normal.movMean,M,1);
        fixPts=fixPts-repmat(normal.fixMean,N,1);

        normal.movScale=sqrt(sum(sum(movPts.^2,2))/M);
        normal.fixScale=sqrt(sum(sum(fixPts.^2,2))/N);

        movPts=movPts/normal.movScale;
        fixPts=fixPts/normal.fixScale;

        nMovLoose(:,1:2)=movPts;
        nFixLoose(:,1:2)=fixPts;
        nMovLoose(:,3)=nMovLoose(:,3)./normal.movScale;
        nFixLoose(:,3)=nFixLoose(:,3)./normal.fixScale;
    end
end