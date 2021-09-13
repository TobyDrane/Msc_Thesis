function screening_index = screenVars(data_discrete,root_index,m,threshold)

% Returns a vector of  indices of the variables (columns) in data_discrete that
% should be included in the analysis.

[numOfobs, numOfvar] = size(data_discrete) ; 


%combine real and noisy data
training_whole = [data_discrete] ; 

% calculate s-distribution
ll = root_index;

for kk = 1:numOfvar
    
    kk;
    
    [s(kk), flag(kk) ] = SignificanceTestByPermutation(training_whole(:, ll), training_whole(:, kk), m, threshold);
    
end


s_record(ll, :) = s ;

s_record(root_index, root_index) = 0  ; 

screening_index= find(s_record(root_index, 1:numOfvar) >= threshold) ; 

