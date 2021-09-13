function [s, flag] = SignificanceTestByPermutation(root, feature, m, threshold)
% TEST whether a given feature should be selected as significant by 
% permutation test. 

if nargin < 2
    display('not enough input data for picking significant test') ; 
elseif nargin < 4
    
    m = 300 ; 
    threshold = 2.6 ; 
end 

% initialization
s = 0 ; 
flag = 0 ; 

% check if the lengths of root and feature are the same
if length(root) ~= length(feature)
    
    display('not equal length for data permutation calculation') ; 
    return ; 
end 

true_mi = mi(root, feature) ; 

numOfobs = length(root) ; 

%calculate the permutated 
for ii = 1:m
    
    perm_mi(ii) = mi(root(randperm(numOfobs)), feature(randperm(numOfobs))) ; 
end 

s = (true_mi - mean(perm_mi))/(std(perm_mi)) ; 

if abs(s) >= threshold
    
    flag = 1 ; 
end 

return ; 

end 