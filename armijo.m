function [alpha_armijo] = armijo(alpha,x,d,f0,f1,gamma,delta)
%% ARMIJO
% DESCRIPTION:
% function to check whether the provided steplength satisfies Armijo 
% condition: f(x+alpha*d)<=f(x)+gamma*alpha*gradient(f(x))'*d
% if this condition is not met, it extracts the greatest value of alpha that
% satisfies Armijo's expression.
%
% function [alpha_armijo] = armijo(alpha,x,d,f0,f1,gamma,delta)
% INPUT:
%       NOTE: (*) indicates necessar y input, the other variables are optional 
%       (*) alpha     - current steplength (1*1);
%       (*) x         - current iterate    (N*1);
%       (*) d         - search direction   (N*1);
%           gamma     - constant provided by the user (1*1) into the range [0,0.5]
%           delta     - constant provided by the user (1*1) into the range [0,  1]
%       (*) f0        - function handle of the objective function          (RN->R );
%       (*) f1        - the gradient (as function handle) of the function  (RN->RN);
% OUTPUT:
%       alpha_armijo - value of alpha whether the condition holds      (1*1);
% REVISION:
%       Ennio Condoleo Rome, Italy h: 23.04 9 Jan 2014
    if (nargin<6)
        delta = 0.5;
        gamma = 1.0e-004;
    elseif (nargin==6)
        delta = 0.5;
    end
    
    j = 1;
    t = 0;
    while (j>0)
        x_new = x+alpha*d;
        if (f0(x_new)<=f0(x)+alpha*gamma*trace(f1(x)'*d))
            j = 0;
            alpha_armijo = alpha;
        else
            alpha = alpha*delta;
            t = t + 1;            
        end    
        if t > 100
            j = 0;
            alpha_armijo = 0;
        end
    end
end