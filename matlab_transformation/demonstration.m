function demonstration(IaColor, IbColor, movLoose, fixLoose, index)

%%=====================================================================
%% $Author: M.E. Su ZHANG, supervised by Assoc. Yang YANG. $
%% $Date: Sat, 30 June 2018$
%% $Contact: sorazcn@gmail.com$
%%=====================================================================
if nargin == 5
    movPtsOrigin = movLoose(index,:);
    fixPts = fixLoose(index,:);
    M = length(index);
    
    Pos1=[movPtsOrigin(:,2) movPtsOrigin(:,1) zeros(M, 1)];
    Pos3=[fixPts(:,2) fixPts(:,1) zeros(M, 1)];
    
    % IaWrapped=tps_warp(IaColor,Pos1,Pos2,'bicubic');
    IaWrapped=tps_warp(IaColor,Pos1,Pos3,'bicubic');
    
    % Show the result
    figure,imshow(IaColor); set(gca,'Position',[0.0 0 1 1]); %title('Source image');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    
    figure,imshow(IbColor); set(gca,'Position',[0.0 0 1 1]); %title('Reference image');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    
    figure,imshow(IaWrapped); set(gca,'Position',[0.0 0 1 1]); %title('Transformed image');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    
    figure,immontage(IbColor, IaWrapped, 8); set(gca,'Position',[0.0 0 1 1]); %title('Checkboard');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    
elseif nargin == 4
    movPtsOrigin = movLoose;
    fixPts = fixLoose;
    M = size(movPtsOrigin,1);
    
    Pos1=[movPtsOrigin(:,2) movPtsOrigin(:,1) zeros(M, 1)];
    Pos3=[fixPts(:,2) fixPts(:,1) zeros(M, 1)];
    
    % IaWrapped=tps_warp(IaColor,Pos1,Pos2,'bicubic');
    IaWrapped=tps_warp(IaColor,Pos1,Pos3,'bicubic');
    
    % Show the result
    figure,imshow(IaColor); set(gca,'Position',[0.0 0 1 1]); %title('Source image');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    
    figure,imshow(IbColor); set(gca,'Position',[0.0 0 1 1]); %title('Reference image');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    
    figure,imshow(IaWrapped); set(gca,'Position',[0.0 0 1 1]); %title('Transformed image');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    
    figure,immontage(IbColor, IaWrapped, 10); set(gca,'Position',[0.0 0 1 1]); %title('Checkboard');
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
end
end

