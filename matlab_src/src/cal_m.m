%======================================================
% Calculate Corresponding Matrix
%======================================================
function [M] = cal_m(source,Nm,target,Nf,T) 
[n, dims] = size (source); %nΪsource����ĵ���
[m, dims] = size (target); %mΪtarget����ĵ���
M0 = ones (n, m);
a00=  zeros (n, m);
%step1 ����source���е����������������㣩
%vS

%step2 ����target���е����������������㣩
%vT


for i=1:dims %ѭ��dims�Σ�dims����ά��
    a0=((source(:,i) * ones(1,m) - ones(n,1) * target(:,i)').^2); %ȫ�־���
    %a0=((vS(:,i) * ones(1,m) - ones(n,1) * vT(:,i)').^2); 
    a000=(source(:,i) * ones(1,m) - ones(n,1) * target(:,i)'); %a000����ƽ��
    %����Local����
    for j=1:size(Nm,2)
    a00=a00+((source(Nm(:,j),i) * ones(1,m)-a000 - ones(n,1) * target(Nf(:,j),i)').^2);
    end
    M0=M0+a0+size(Nm,2)^2*T*(a00); %��Ͼ�����㣨Ҳ���ǰ�global��local������һ��
    a00=0;
end

if n==m % for non outlier case
    M=round(M0*1e6);
    M=lap(M);
else % for outlier case
    M=round(M0*1e6);
    M=lap_wrapper(M,1e9);
end

end