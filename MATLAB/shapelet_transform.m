clear
close all

%% Load learned shapelets
directory = '../JMLToolkit/UCR_TS_Archive_2015';
tsName = 'MIMICIII_infection_NEG_VS_shock_pos_POS_60age_1wks_2sub_concatenate';

fName = dir([directory '/' tsName '/' tsName '_LearnedShapelets*']);

n_shapelet_file = length(fName)-1;
sl = cell(n_shapelet_file,1);

for k = 1:n_shapelet_file
    temp = readtable([directory '/' tsName '/' fName(k).name],...
        'ReadVariableNames',false,'Delimiter',' ');
    temp  = table2array(temp);
    sl{k} = temp(:,3:end-1);
end

%% Load training and testing time series data
ts_train = readtable([directory '/' tsName '/' tsName '_TRAIN'],...
        'ReadVariableNames',false,'Delimiter',',');

y_train = table2array(ts_train(:,1))-1; % class labels are 0 and 1
ts_train = table2array(ts_train(:,2:end));
 
n_train = size(ts_train,1);

ts_test = readtable([directory '/' tsName '/' tsName '_TEST'],...
        'ReadVariableNames',false,'Delimiter',',');

y_test = table2array(ts_test(:,1))-1;   % class labels are 0 and 1
ts_test = table2array(ts_test(:,2:end));
n_test = size(ts_test,1);

%%
R = length(sl);    % number of groups of shapelets
K = size(sl{1},1); % number of shapelets in each group

tf_train = nan(R,K,n_train);
tf_test = nan(R,K,n_test);

for r = 1:R
    for k = 1:K
        for i = 1:n_train
            [tf_train(r,k,i),~] = time_series_dist(sl{r}(k,:),ts_train(i,:));
        end
    end
end
x_train = reshape(tf_train,[],n_train)';

for r = 1:R
    for k = 1:K
        for i = 1:n_test
            [tf_test(r,k,i),~] = time_series_dist(sl{r}(k,:),ts_test(i,:));
        end
    end
end
x_test = reshape(tf_test,[],n_test)';

save([directory '/' tsName '/' tsName '_TRAIN_TF'],'tf_train','y_train','x_train');
save([directory '/' tsName '/' tsName '_TEST_TF'],'tf_test','y_test','x_test');
