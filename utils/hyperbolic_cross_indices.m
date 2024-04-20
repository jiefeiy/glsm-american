function a = hyperbolic_cross_indices(d, k)
% Hyperbolic cross indices
%
% Produces all indices a \in \N_0^d that satisfy the condition
%
%   \prod_{j=1}^d (a_j + 1) \leq k + 1.

if d==1
  a = (0:k).';
  return
end

a = zeros([1 d]);

% First add all indices with sparsity 1
for q = 1:d
    temp = zeros([k d]);
    temp(:,q) = 1:k;
    a = [a; temp];
end

% Now determine the maximum 0-norm the entries can be. I.e., for which values
% of p is 2^p <= k+1?
pmax = floor(log(k+1)/log(2));

% For each sparsity p, populate with all possible indices of that sparsity
for p = 2:pmax
    % Determine all possible locations where nonzero entries can occur
    combs = nchoosek(1:d, p);
    
    % Now we have 2^p < k+1, i.e., an index with nonzero entries ones([p 1]) is ok. 
    % Keep incrementing these entries until product exceeds k+1
    possible_indices = ones([1 p]);
    ind = 1;
    % This is slow and stupid, but should only have complexity log(k)-ish, so
    % it's probably ok.
    while ind <= size(possible_indices, 1)
        % Add any possibilities that are children of possible_indices(ind,:)
        alph = possible_indices(ind,:);
        
        for q = 1:p
            temp = alph;
            temp(q) = temp(q) + 1;
            if prod(temp+1) <= k+1
                possible_indices(end+1,:) = temp;
            end
        end
        
        ind = ind + 1;
    end
    
    possible_indices = unique(possible_indices, 'rows');
    
    arow = size(a,1);
    a = [a; zeros([size(combs,1)*size(possible_indices, 1) d])];
    
    % Now for each combination, we put in possible_indices
    for c = 1:size(combs, 1)
        i1 = arow + 1;
        i2 = arow + size(possible_indices, 1);
        
        a(i1:i2,combs(c,:)) = possible_indices;
        
        arow = i2;
    end

end
