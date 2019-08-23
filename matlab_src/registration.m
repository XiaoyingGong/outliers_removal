function result = registration(test1, test2)
    Mpoints = test1;
    Fpoints = test2;
    FlattenedData = Mpoints(:)'; % 展开矩阵为一列，然后转置为一行。
    MappedFlattened = mapminmax(FlattenedData, 0, 1); % 归一化。
    Mpoints = reshape(MappedFlattened, size(Mpoints));

    FlattenedData = Fpoints(:)'; % 展开矩阵为一列，然后转置为一行。
    MappedFlattened = mapminmax(FlattenedData, 0, 1); % 归一化。
    Fpoints = reshape(MappedFlattened, size(Fpoints));

    % Calculate T_init and T_final
    sizeM=size(Mpoints,1);
    sizeF=size(Fpoints,1);
    dis=zeros(sizeM,sizeF);
    Tmax=zeros(sizeM,sizeM);
    for i=1:3
        dis = dis + (Mpoints(:,i) * ones(1,sizeF) - ones(sizeM,1) * Fpoints(:,i)').^2;
        Tmax = Tmax + (Mpoints(:,i) * ones(1,sizeM) - ones(sizeM,1) * Mpoints(:,i)').^2;
    end
    T=sqrt(max(max(dis)))/10;
    for i=1:sizeM
        [v,ind]=min(Tmax(i,:));
        Tmax(i,ind)=1000;
    end
    T_final=sum(min(Tmax'))/((sizeM)*8);

    % Initial Parameters For display
    Source=Mpoints;
    Target=Fpoints;

    % Input Data
    x= Mpoints; % source point-set
    y= Fpoints; % target point-set
    xw = x;     % Initial x^w

    % Annealing Parameter
    lambda_init = length(x);
    anneal_rate = 0.7;

    % Set 5 closest neighbor points
    K=5;
    Nm=findN(Mpoints,K);% NmΪ��������5���ľ��󣬵�N�б�ʾ��N������������ţ�
    Nf=findN(Fpoints,K);

    % Other parameter initialzations
    acc=0;       % Error
    flag_stop=0; % Stop fig
    btn=0;       % for mouse click
    step=0;
    lambda=lambda_init*length(x)*T; % Initial lambda

    while (flag_stop ~= 1)
        if T <T_final
           flag_stop =1;
        end
        %========================================================
        % A Global and Local Distance-based Point Mathicng Method
        %========================================================
    %     if btn==2
            % Calculate two-way corresponding matrix M
            [m]=cal_m(xw,Nm,y,Nf,T);

            % Update the correspondence x^c for the source point-set x
            xc=m*y;

            % Update TPS transformation
            lambda = lambda_init*T; %non-rigid warping
            [w,d,k]  = update_tps(x, xc, lambda);

            % Update the warping template x^w
            xw = update_xw(x,w,d,k);

            % Reduce T
            T  = T * anneal_rate;
            Mpoints=xw; % Output
    %     end
    end
    result = Mpoints;
    FlattenedData = result(:)'; % 展开矩阵为一列，然后转置为一行。
    MappedFlattened = mapminmax(FlattenedData, 0, 800); % 归一化。
    result = reshape(MappedFlattened, size(result));
end
