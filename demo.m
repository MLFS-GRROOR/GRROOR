% This is an example file on how the GRROOR [1] program could be used.

% Global Redundancy and Relevance Optimization in Orthogonal Regression for Embedded Multi-label Feature Selection, In Proc. 30th Int. Joint Conf. Artif. Intell, 2021 (Under review).

% Please feel free to contact me (bmexuxueyuan@163.com), if you have any problem about this program.

clc; clear; 
addpath(genpath('.\'))
load('education.mat')
X_train = train_data;
Y_train = train_target';
X_test = test_data;
Y_test =  test_target';

% Calucate some statitics about the data
[num_train, num_label] = size(Y_train); [num_test, num_feature] = size(X_test);

pca_remained = round(num_feature*0.95);

% Performing PCA
all = [X_train; X_test]; 
ave = mean(all);
all = (all'-concur(ave', num_train + num_test))';

covar = cov(all); covar = full(covar);

[u,s,v] = svd(covar);

t_matrix = u(:, 1:pca_remained)';
all = (t_matrix * all')';

X_train = all(1:num_train,:); X_test = all((num_train + 1):(num_train + num_test),:);

% Experimental Setting
Y_train(Y_train==0) = -1;
Y_test(Y_test==0) = -1;    

X = X_train';
Y = Y_train';

po_fir = 1;
po_step = 1;
po_end = 50;
    
alpha = 1; beta = 0.8; lambda = 0.6; c =2; cycle = 10;

% Running the GRROOOR procedure for feature selection
t0 = clock;
[weight,feature_id]= GRROOR(X,Y,lambda,alpha,beta,c,cycle);
time = etime(clock, t0);

feature_idx = feature_id;

% The default setting of MLKNN
Num = 10;Smooth = 1;  

% Train and test
% If you use MLKNN as the classifier, please cite the literature [2]
% [2] M.-L. Zhang, Z.-H. Zhou:
% ML-KNN: A lazy learning approach to multi-label learning. Pattern Recognition 2007, 40(7): 2038-2048.
for i = po_fir : po_step : po_end
    fprintf('Running the program with the selected features - %d/%d \n',i,po_end);
    
    f=feature_idx(1:i);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(X_train(:,f),Y_train',Num,Smooth);
    [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
        MLKNN_test(X_train(:,f),Y_train',X_test(:,f),Y_test',Num,Prior,PriorN,Cond,CondN);
    
    HL_GRROOR(i)=HammingLoss;
    RL_GRROOR(i)=RankingLoss;
    CV_GRROOR(i)=Coverage;
    AP_GRROOR(i)=Average_Precision;
    MA_GRROOR(i)=macrof1;
    MI_GRROOR(i)=microf1;
end
