% ALM algotithm
% min theta'*A*theta-theta'b
% theta'*1=1,theta>=0
% the aim is to get theta

function [theta,v]=ALM(A,b)
% input:
% A as the matrix with dimension d*d;
% b as the colume vector with d*1;
% output:
% theta as the colume vector with d*1 
% the sum of elements is 1,and every element is positive;

d=size(A,1);
rho=1.01;
v=zeros(d,1);
theta=1/d*ones(d,1);
%v=rand(d,1);
%theta=rand(d,1);
lambda1=zeros(d,1);
lambda2=0;
mu=0.1;
Id=eye(d);
yid=ones(d,1);

ITER = 100;
% obj2 = zeros(ITER,1);
% obj2(1)=theta'*A*theta-theta'*b;
err=1;t=1;
for t = 1:ITER
% while err>1e-3
    E=2*A+mu*Id+mu*ones(d);
    f=mu*v+mu*yid+b-lambda2*yid-lambda1;
    theta=(E)\f;
    q=theta+1/mu*lambda1;
    v=max(q,0);
    lambda1=lambda1+mu*(theta-v);
    lambda2=lambda2+mu*(theta'*yid-1);
    mu=rho*mu;
    obj2(t)=theta'*A*theta-theta'*b;
    if t>1
         change=abs((obj2(t)-obj2(t-1))/obj2(t));
         if change<1e-8
             break;
         end
    end   
% %     if t>=2
% %         err=abs(obj2(t-1)-obj2(t)/obj(iter));
% %     end
% %     t=t+1;
end
plot(obj2);
grid on;
ylabel('Objectve value');
xlabel('Iteration');
xlim([1,ITER]);
title('ALM method');
end