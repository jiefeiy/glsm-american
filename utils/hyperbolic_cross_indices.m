function a = hyperbolic_cross_indices(d, k)
% Reference:
% [1] : "Computation of Induced Orthogonal Polynomial Distributions", Akil Narayan,...
% arXiv:1704.08465 [math], http://arxiv.org/abs/1704.08465
% Github repo: https://github.com/akilnarayan/induced-distributions/tree/master

if d==1
  a = (0:k).';
  return
end
a = zeros([1 d]);
for q = 1:d
    temp = zeros([k d]);
    temp(:,q) = 1:k;
    a = [a; temp];
end
pmax = floor(log(k+1)/log(2));

for p = 2:pmax
    combs = nchoosek(1:d, p);
    possible_indices = ones([1 p]);
    ind = 1;
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
    
    for c = 1:size(combs, 1)
        i1 = arow + 1;
        i2 = arow + size(possible_indices, 1);
        a(i1:i2,combs(c,:)) = possible_indices;
        arow = i2;
    end
end
