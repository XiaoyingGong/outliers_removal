%% һЩ������������
function config = initilizationSIR

%%=====================================================================
%% $Author: M.E. Su ZHANG, supervised by Assoc. Yang YANG. $
%% $Date: Sat, 30 June 2018$
%% $Contact: sorazcn@gmail.com$
%%=====================================================================
    config.lambda = 0.5;            % TPS smoothness
    config.K=5;                         % Number of nearest inliers
    config.Radial=50;     
    config.Theta=120;               % Radial and tangential directions of CALM 
    config.thresDist =0.01;        % Threshold for pruning
    config.thresHistDiff =2.5;   %Threshold for CALM
    config.viz=0;                       % set to 1 for visualization.
end

