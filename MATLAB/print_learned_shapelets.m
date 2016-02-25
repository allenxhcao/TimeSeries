clear
close all

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

%% Load training data
ts_train = readtable([directory '/' tsName '/' tsName '_TRAIN'],...
        'ReadVariableNames',false,'Delimiter',',');

ts_train = table2array(ts_train(:,2:end));

%% 
short = sl{3}; % whole bunch of short time series
long = ts_train(3,:); % one long time series

nsl = size(short,1);

nrow = floor(sqrt(nsl));
ncol = ceil(sqrt(nsl));
figure('Units','pixels','Position',[0 0 nrow*200 ncol*200])
for k = 1:nsl
    subplot(nrow,ncol,k)
    hold on
    [dist,ind] = time_series_dist(short(k,:),long);
    plot(1:length(long),long);
    plot(ind:ind+length(short(k,:))-1,short(k,:));
    xlim([1 length(long)])
    hold off
    title(num2str(k))
end