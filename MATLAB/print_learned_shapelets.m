clear
close all

dir = '../JMLToolkit/UCR_TS_Archive_2015';
tsName = 'MIMICIII_normal_AS_NEG_VS_shock_pos_AS_POS_mean';

sl = readtable([dir '/' tsName '/' tsName '_LearnedShapelets'],...
    'ReadVariableNames',false,'Delimiter',' ');

sl  = table2array(sl);

K = sum(sl(:,1)==0);
R = floor(sqrt(size(sl,1)/K*2));
Q = size(sl,2)-2-1;
lsl = cell(R,1);

rowCount = 1;
for r = 1:R
    lsl{r} = nan(K,r*size(sl,2)-2-1-(r-1)*R);
    for k = 1:K
        sltemp = reshape(sl(rowCount:rowCount+r-1,:)',1,[]);
        lsl{r}(k,:) = sltemp(3:end-1-(r-1)*R);
        rowCount = rowCount + r;
    end

end

for r = 1:R
    subplot(R,1,r)
    plot(lsl{r}');
    xlim([1 R*Q])
end

short = lsl{1};
long = lsl{3}(2,:);

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