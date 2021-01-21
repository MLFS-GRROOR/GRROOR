 %Global Redundancy and Relevance Optimization in Orthogonal Regression for Embedded Multi-label Feature Selection
%
function [weight,feature_id]= GRROOR(X,Y,lambda,alpha,beta,c,cycle)
%Input: 
% X as the data matrix with dimension d*n; Y as the label matrix with dimension k*n;
% d is the dimension of the feature,n is the numbers of the samples;
% k is the numbers of the class;
% parameters: lambda,alpha,beta,c,cycle.
% cycle is cycle index.

%Output: 
% weight and feature_id.
% weight as the score (descending sort) of features with dimension 1*d;
% feature_id as feature index (descending sort) with dimension 1*d; 


[d,n] = size(X);
[k,n] = size(Y);

In=eye(n);
yi=ones(n,1);
H=In-yi*yi'/n;
B = rand(k,c);
V = rand(c,n);

i=1/d*ones(1,d);
THETA=diag(i);  %fix THETA;  
ITER = cycle;
%obj=zeros(ITER,1);

[RM]=  GloRed_eva(X,H,d);
ind_RM =find(isnan(RM)); RM(ind_RM)=0;

options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 1;
AGM = constructW(X',options);
vec_ADM = sum(AGM,2); 
ADM = diag(vec_ADM);
GLM = ADM - AGM;

% The relevance betwwen labels
LL_C = pdist2( Y, Y, 'cosine' );
ind=find(isnan(LL_C)); LL_C(ind)=0; LL = 1-LL_C;

% iter = 1;

for iter=1:ITER
    C=THETA*X*H*X'*THETA';
    D=THETA*X*H*V';
    fprintf('this is the %d GPI iteration\n',iter);
    W=GPI(C,D,1);   
   
    Q=(X*H'*X').*(W*W')+lambda*RM; 
    S=2*X*H*V'*W';
    s=diag(S);
    
    fprintf('this is the %d ALM iteration\n',iter);
    [theta,v]=ALM(Q,s);    
    THETA=diag(theta'); 
   
    d_V = 2*(alpha*B'*(B*V-Y)+ V*GLM + (V*H - W'*THETA*X*H)*H'); 
    d_B = 2*alpha*(B*V-Y)*V'+2*beta*LL*B;

    M = W'*THETA*X*H-V*H;
    ML = Y-B*V;
    
    f0_V = @(V)norm(M,'fro')*norm(M,'fro') + trace(V*GLM*V') +lambda*theta'*RM*theta + alpha*norm(ML,'fro')*norm(ML,'fro')+ beta*trace(LL*B*B');
    f1_V = @(V)2*(alpha*B'*(B*V-Y)+ V*GLM + (V*H - W'*THETA*X*H)*H'); 
    lambda_V = armijo(1,V,-d_V,f0_V,f1_V);
    lambda_V_test(iter) = lambda_V;
    if iter >1 && lambda_V_test(iter)==1 && lambda_V_test(iter-1)~= 1
        break;
    end
    
    f0_B = @(B)norm(M,'fro')*norm(M,'fro') + trace(V*GLM*V') +lambda*theta'*RM*theta + alpha*norm(ML,'fro')*norm(ML,'fro')+ beta*trace(LL*B*B');
    f1_B = @(B)2*alpha*(B*V-Y)*V'+2*beta*LL*B;   
    lambda_B = armijo(1,B,-d_B,f0_B,f1_B);
    lambda_B_test(iter) = lambda_B;
    if iter >1 && lambda_B_test(iter)==1 && lambda_B_test(iter-1)~= 1
        break;
    end
    
    V = V - lambda_V*d_V;    
    B = B - lambda_B*d_B; 
    
    M = W'*THETA*X*H-V*H;
    ML = Y-B*V;
    obj(iter)=norm(M,'fro')*norm(M,'fro') + trace(V*GLM*V') +lambda*theta'*RM*theta + alpha*norm(ML,'fro')*norm(ML,'fro')+ beta*trace(LL*B*B');
%      if iter>1
%          change=abs((obj(iter)-obj(iter-1))/obj(iter));
%          if change<1e-8
%              break;
%          end
%      end   
end
[weight_norm,ps] = mapminmax(theta',0,1);
theta = weight_norm/sum(weight_norm);
[weight,feature_id]=sort(theta,'descend'); 
plot(obj,'*-r');
ylabel('Objectve value');
xlabel('Iteration');
xlim([1,ITER]);
title('GRROOR method');