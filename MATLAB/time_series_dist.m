function [dist, ind] = time_series_dist(short,long)

if length(short)>length(long);
    temp = short;
    short = long;
    long = temp;
end

l1 = length(short);
l2 = length(long);

dist_temp = zeros(12-l1+1,1);

for k = 1:length(1:l2-l1+1)
    dist_temp(k) = 1/l1*sum((short - long(k:k+l1-1)).^2);
end

[dist, ind] = min(dist_temp);