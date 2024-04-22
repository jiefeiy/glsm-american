function [A, G] = gen_poly_basis_grad(type, I, grid, scale)
% grid should be a column vector
[d,N] = size(I);       % get N (number of matrix columns) and d (dimension)
M = size(grid,1);      % get m (number of matrix rows)
A = zeros(M,N);        % initialize A
order = max(I(:));     % find maximum polynomial degree
P1 = cell(1,d);        % store 1d basis

if nargout > 1
    Pd1 = cell(1,d);       % store 1d derivative
end

% first dim use norm_hermite 
t = scale{1};
yy = grid(:,1);
P1{1} = generate_poly_basis_1d(type{1}, order, yy, t);   % M-by-(order+1) matrix
if nargout > 1
    Pd1{1} = generate_poly_grad_1d(type{1}, order, yy, t);   % M-by-(order+1) matrix
end

% second dim use chebyshev
domain = scale{2};
yy = grid(:,2);
P1{2} = generate_poly_basis_1d(type{2}, order, yy, domain); 
if nargout > 1
    Pd1{2} = generate_poly_grad_1d(type{2}, order, yy, domain);   % M-by-(order+1) matrix
end

% assemble 2d basis by tensor product
for n = 1:N
    P_all = zeros(M,d);
    for j = 1:d
        P_all(:,j) = P1{j}(:, I(j,n)+1);
    end
    A(:,n) = prod(P_all,2);
end

if nargout > 1
    G = cell(1,d);
    for j = 1:d
        G{j} = zeros(M,N);
        for n = 1:N
            Q_all = zeros(M,d);
            for k = 1:d
                if k == j
                    Q_all(:,k) = Pd1{k}(:, I(k,n)+1);
                else
                    Q_all(:,k) = P1{k}(:, I(k,n)+1);
                end
            end
            G{j}(:,n) = prod(Q_all,2);
        end
    end
    clear Pd1;           % release memory
end
clear P1;

end

function G = generate_poly_grad_1d(type, order, grid, scale)
G = zeros(numel(grid), order+1);  
if isequal(type, 'chebyshev')
    %%% gradient of the Chebyshev polynomials of the first kind T_n(x)
    % T_n(x) = cos( n acos(x) )  
    % d T_n(x) / dx = n U_{n-1}(x), ==> G(:,n+1)
    % where     U_{n-1}(x) = 2x U_{n-2}(x) - U_{n-3}(x) 
    % or        U_{n-1}(x) = sin( n * theta ) ./ sin(theta) 
    % with           theta = acos(x).
    domain = scale;
    xmin = domain(1); xmax = domain(2);
    grid(grid < xmin) = xmin;    grid(grid > xmax) = xmax;
    grid = (grid - xmin) * 2/(xmax - xmin) - 1;    % shift to [-1,1]
    U = zeros(length(grid), order);
    U(:,1) = 1;  U(:,2) = 2*grid;
    G(:,2) = 1; 
    G(:,3) = 4*grid;
    for i = 3:order
        U(:,i) = (2*grid.*U(:,i-1) - U(:,i-2));
        G(:,i+1) = i * U(:,i);
    end
    idx = (grid < -1) | (grid > 1);
    G(idx,:) = 0;
elseif isequal(type, 'hermite')
    %%% gradient of the probabilist's Hermite polynomials
    % Hn'(x) = n H_{n-1}(x) 
    %%% gradient of the generalized Hermite polynomials
    A = zeros(numel(grid), order);
    t = scale; 
    yy = grid/sqrt(t);
    A(:,1) = 1; A(:,2) = yy;
    G(:,2) = 1; G(:,3) = 2*yy;
    for i = 3:order
        A(:,i) = yy .* A(:,i-1) - (i-2) * A(:,i-2);
        G(:,i+1) = i * A(:,i);
    end
    G = G .* ( t.^(([0 0:order-1])/2) );
elseif isequal(type, 'norm_hermite')
    %%% gradient of normalized generalized Hermite polynomials
    % H_n^t(x) * t^(-n/2) * (n!)^(-.5).
    A = zeros(numel(grid), order);
    t = scale; 
    yy = grid/sqrt(t);
    A(:,1) = 1; A(:,2) = yy;
    G(:,2) = 1; G(:,3) = 2*yy;
    for i = 3:order
        A(:,i) = yy .* A(:,i-1) - (i-2) * A(:,i-2);
        G(:,i+1) = i * A(:,i);
    end
    G = G/sqrt(t);
    G = G ./ sqrt(factorial(0:order));
else
    fprintf('wrong type');
end

end

