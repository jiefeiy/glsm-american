function A = generate_poly_hermite(type, I, grid, scale)
% grid should be a column vector
[N,d] = size(I);       % get N (number of matrix columns) and d (dimension)
M = size(grid,1);      % get m (number of matrix rows)
A = zeros(M,N);        % initialize A
order = max(I(:));     % find maximum polynomial degree
P1 = cell(1,d);        % store 1d basis

for j = 1:d
    yy = grid(:,j);
    P1{j} = generate_poly_basis_1d(type, order, yy, scale);   % M-by-(order+1) matrix
end

% assemble d-dim basis by tensor product
parfor n = 1:N
    P_all = zeros(M,d);
    for j = 1:d
        P_all(:,j) = P1{j}(:, I(n,j)+1);
    end
    A(:,n) = prod(P_all,2);
end

end



