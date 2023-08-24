%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author: Junlong Feng | HKUST | Feb 2023
%%% Example code to implement Algorithm 1 in "Nuclear Norm Regularized
%%% Quantile Regression with Interactive Fixed Effects" by Junlong Feng.
%%% DGP follows the Monte Carlo section in the paper.
%___________________________________________________________________________
clear;clc

N=300;T=300;
tau=0.7 % quantile level
p=10; % number of regressors

btrue=[-1+0.1*tau;1+0.1*tau]; % true coefficients at tau
btrue=repmat(btrue,5,1); 

lambda=sqrt(log(N*T))*sqrt(max(N,T))/(4*N*T);
n_iter=100  ;


sc=parallel.pool.Constant(RandStream('Threefry')) % set up parallel computing
parfor B=1:n_iter
stream=sc.Value;
stream.Substream=2022+B;

%% DGP
F=[2*rand(stream,T,1),2*rand(stream,T,1),2*rand(stream,T,1),2*rand(stream,T,1),2*rand(stream,T,1)];
xi=[2*rand(stream,N,1),2*rand(stream,N,1),2*rand(stream,N,1),2*rand(stream,N,1),2*rand(stream,N,1)];

eta1=2*rand(stream,N,T);
eta2=1*rand(stream,N,T);
eta3=2*rand(stream,N,T);
eta4=1*rand(stream,N,T);
eta5=2*rand(stream,N,T);
eta6=1*rand(stream,N,T);
eta7=2*rand(stream,N,T);
eta8=1*rand(stream,N,T);
eta9=2*rand(stream,N,T);
eta10=rand(stream,N,T); 

x1=zeros(N,T);
x2=zeros(N,T);
x3=zeros(N,T);
x4=zeros(N,T);
x5=zeros(N,T);

phi=0.2; 
for i=1:N
    for t=1:T
        x1(i,t)=eta1(i,t)+phi*F(t,1)^2+phi*xi(i,1)^2;
        x2(i,t)=eta2(i,t)+phi*F(t,2)^2+phi*xi(i,2)^2;
        x3(i,t)=eta3(i,t)+phi*F(t,3)^2+phi*xi(i,3)^2;
        x4(i,t)=eta4(i,t)+phi*F(t,4)^2+phi*xi(i,4)^2;
        x5(i,t)=eta5(i,t)+phi*F(t,5)^2+phi*xi(i,5)^2;
    end
end

x6=eta6+0.2*x1;        
x7=eta7+0.2*x2;       
x8=eta8+0.2*x3;        
x9=eta9+0.2*x4;        
x10=eta10+0.2*x5;       

Y=zeros(N,T);
Ltrue=zeros(N,T);
u=rand(stream,N,T);
e=icdf('Normal',u,0,1);   %% uncomment for normal 
e0=icdf('Normal',tau,0,1);
% e=icdf('T',u,2);  %% uncomment for student's t
% e0=icdf('T',tau,2);
       
for i=1:N
    for t=1:T
        Y(i,t)=(-1+0.1*u(i,t))*(x1(i,t)+x3(i,t)+x5(i,t)+x7(i,t)+x9(i,t))+(1+0.1*u(i,t))*(x2(i,t)+x4(i,t)+x6(i,t)+x8(i,t)+x10(i,t))+F(t,1)*xi(i,1)+F(t,1)*0.1*u(i,t)+F(t,2)*xi(i,2)+F(t,2)*0.1*u(i,t)+F(t,3)*xi(i,3)+F(t,3)*0.1*u(i,t)+(0.35<u(i,t))*(F(t,4)*xi(i,4)+F(t,4)*0.1*u(i,t))+(0.65<u(i,t))*(F(t,5)*xi(i,5)+F(t,5)*0.1*u(i,t))+e(i,t)+2;
        Ltrue(i,t)=F(t,1)*xi(i,1)+F(t,1)*0.1*tau+F(t,2)*xi(i,2)+F(t,2)*0.1*tau+F(t,3)*xi(i,3)+F(t,3)*0.1*tau+(tau>0.35)*(F(t,4)*xi(i,4)+F(t,4)*0.1*tau)+(tau>0.65)*(F(t,5)*xi(i,5)+F(t,5)*0.1*tau)+e0+2;
    end
end

%% Estimation
% Initialize parameters. b0 and b1 are betas of the current and the next
% steps. L0 & L1, V0 & V1 have similar meanings.
b0=zeros(p,1);
b0_inner=zeros(p,1);
b1=b0;
L0=zeros(N,T);
L1=zeros(N,T);
V0=zeros(N,T);
V1=V0;
mu=(N*T)/(4*sum(sum(abs(Y)))); % Following Algorithm 1
H=zeros(N,T);
delta=1; % increment of the outer loop


X_regressor=[ones(N*T,1),x1(:),x2(:),x3(:),x4(:),x5(:),x6(:),x7(:),x8(:),x9(:),x10(:)];

% Implement Algorithm 1
tic
M=Y-reshape(X_regressor(:,2:end)*b0,N,T);
while  delta>1e-6 % outer loop
    [U,Sigma,V]=svd(M-V0+H/mu,"econ"); 
    Sigma=Sigma-diag(ones(min(N,T),1))/mu; % equation (A.6)
    Sigma=(Sigma>0).*Sigma;
    L1=U*Sigma*V';        
    delta2=1; % start the inner loop
    while delta2>1e-4
        % equation (A.7). M-L1+H/mu is Gamma_V.
        V1=sign(M-L1+H/mu).*max(abs(M-L1+H/mu)-((M-L1+H/mu>0)*tau+(M-L1+H/mu<0)*(1-tau))/(mu*lambda*N*T),zeros(N,T));
        % equation (A.8). dep here is vec(Gamma_beta)
        dep=H/mu+Y-V1-L1;
        dep=dep(:);
        btemp=lsqr(X_regressor,dep); % solve the linear equation X'beta=vec(Gamma_beta)
        b1=btemp(2:end); %Drop the constant.
        delta2=mean((b1-b0_inner).^2); 
        b0_inner=b1;
        M=Y-reshape(X_regressor(:,2:end)*b1,N,T);
    end
    delta=mean((b1-b0).^2)+sum(sum((L1-L0).^2))/(N*T);
    L0=L1;
    V0=V1;
    b0=b1;
    H=H+mu*(M-V1-L1); % equation (A.5)
end

timeNuclear(B,1)=toc;

% For MSE_L
resid1(B,1)=sum(sum((Ltrue-L0).^2))/(N*T);

% For MSE_q
resid2(B,1)=sum(sum((reshape(X_regressor(:,2:end)*b0,N,T)-reshape(X_regressor(:,2:end)*btrue,N,T)+L0-Ltrue).^2))/(N*T);

% For MaxDev_L
resid3(B,1)=max(abs(Ltrue(:)-L0(:)));

bhat(:,B)=b0;

end
bias=sum(bhat,2)/n_iter-btrue; % bias for each regressor
avbias=bias'*bias/length(btrue); % average bias squared.
var=(diag(bhat*bhat'))/n_iter-mean(bhat,2).^2; % variance for each regressor
avvar=mean(var); % average variance
avtime=mean(timeNuclear); % average computation time
MSEL=mean(resid1); 
MSEq=mean(resid2);
MaxDevL=mean(resid3);
