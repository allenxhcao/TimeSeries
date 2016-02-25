clear
close all

addpath(genpath('C:\Users\xi\Box Sync\xlibrary\MATLAB'))
addpath(genpath('C:\Users\XiHang\Box Sync\xlibrary\MATLAB'))


%% Load training and testing data
directory = '../JMLToolkit/UCR_TS_Archive_2015';
tsName = 'MIMICIII_infection_NEG_VS_shock_pos_POS_60age_1wks_2sub_concatenate';

load([directory '/' tsName '/' tsName '_TRAIN_TF']);
load([directory '/' tsName '/' tsName '_TEST_TF']);

%% Normalize data
% [x_train_normalized, setting] = ...
%     normalize_lr(x_train,ones(size(x_train,1),1));
% x_test_normalized = ...
%     normalize_lr(x_test,ones(size(x_test,1),1),setting);
% 
% %% Learning and evaluation
% theta = logiRegr(x_train_normalized,y_train);
% y_pred = logiEval(theta,x_test_normalized);
% perf= binary_classification_performance_evaluation(y_test,y_pred);

%%
weight = readtable([directory '/' tsName '/' tsName '_weights'],...
        'ReadVariableNames',false,'Delimiter',' ');
    
weight = table2array(weight);
R = size(weight,1)/2;
K = size(weight,2)-1;

w = zeros(R*K+1,2);
w(1,1) = weight(1,end);
w(1,2) = weight(R+1,end);

weight(:,end) = [];
w(2:end,1) = reshape(weight(1:R,:),[],1);
w(2:end,2) = reshape(weight(R+1:R+R,:),[],1);

y_pred = logiEval(w(:,2),x_test);
perf= binary_classification_performance_evaluation(y_test,y_pred);


