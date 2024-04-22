function A = generate_poly_hermite_par(type, I, grid, scale, num_par)
% grid should be a column vector
% num_par is the number of cores available for parallel computing
[N,d] = size(I);       % get N (number of matrix columns) and d (dimension)
M = size(grid,1);      % get m (number of matrix rows)
mm = M/num_par;
A = cell(num_par, 1);
order = max(I(:));     % find maximum polynomial degree
parfor c = 1:num_par
    P1 = cell(1,d);        % store 1d basis
    for j = 1:d
        yy = grid( (mm*(c-1)+1):(mm*c), j );
        P1{j} = generate_poly_basis_1d(type, order, yy, scale);   % M-by-(order+1) matrix
    end
    yy = [];
    
    % assemble d-dim basis by tensor product
    A_slice = zeros(mm, N);
    for n = 1:N
        P_all = zeros(mm,d);
        for j = 1:d
            P_all(:,j) = P1{j}(:, I(n,j)+1);
        end
        A_slice(:,n) = prod(P_all,2);
    end
    P1 = [];
    P_all = [];   % release memory 
    A{c} = A_slice;
end
A = cell2mat(A);

end