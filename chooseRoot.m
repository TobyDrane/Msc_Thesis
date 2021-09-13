function root_index = chooseRoot(data_discrete,noisy_magnitude,m,threshold)

[numOfobs, numOfvar] = size(data_discrete) ; 

%create noisy data 
for ll = 1:numOfvar
    for nn = 1:noisy_magnitude

        noisy_training(:, noisy_magnitude*(ll-1)+nn) = data_discrete(randperm(numOfobs), ll) ; 
    end 
end 


%combine real and noisy data
training_whole = [data_discrete noisy_training] ; 

% calculate s-distribution
for ll = 1:numOfvar

    ll;

    for kk = 1:(numOfvar*(noisy_magnitude+1))

        kk;

        [s(kk), flag(kk) ] = SignificanceTestByPermutation(training_whole(:, ll), training_whole(:, kk), m, threshold); 

    end 

    % calculate the p-values by comparing the s-distributions for real
    % and noisy data
    s_record(ll, :) = s ; 

    s(ll) = [] ; % remove the s for the root itself. 

    [h(ll), p(ll)] = ttest2(s(1:(numOfvar-1)), s(numOfvar:end), 0.05, 'both', 'unequal') ; % test with original s-values

%         [h(ii, jj), p(ii, jj)] = ttest2(abs(temp(1:36)), abs(temp(37:end)), 0.05, 'both', 'unequal') ; % test with the absolute values of  s

end 

% pick the node with the smallest p-value as the root
[min_value_p, root_index] = min(p) ;

% after record the selected root in root_index, 
s_record(root_index, root_index) = 0  ; 



