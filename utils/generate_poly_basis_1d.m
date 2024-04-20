function A  = generate_poly_basis_1d(type, order, grid, scale)
A = zeros(numel(grid), order + 1);
if isequal(type, 'chebyshev')
    %%% Chebyshev polynomials of the first kind T_n(x)
    % T_n(x) = cos( n acos(x) )  ==> A(:,n+1)
    domain = scale;
    xmin = domain(1); xmax = domain(2);
    % grid(grid < xmin) = xmin;    grid(grid > xmax) = xmax;
    grid = (grid - xmin) * 2/(xmax - xmin) - 1;    % shift to [-1,1]
    A(:,1) = 1;
    for i = 1:order
        A(:,i+1) = cos(i.*acos(grid));
    end
    idx = (grid < -1) | (grid > 1);
    A(idx,:) = 0;
elseif isequal(type, 'hermite')
    %%% probabilist's Hermite polynomials 
    % H_0(x) = 1, H_1(x) = x. 
    % For n >= 2, H_n(x) = x H_{n-1}(x) - (n-1) H_{n-2}(x).
    %%% generalized Hermite polynomials
    % H_n^t(x) = t^(n/2) H_n( x/sqrt(t) ).
    t = scale; 
    yy = grid/sqrt(t);
    A(:,1) = 1; A(:,2) = yy;
    for i = 2:order
        A(:,i+1) = yy .* A(:,i) - (i-1) * A(:,i-1);
    end
    A = A .* ( t.^((0:order)/2) );
elseif isequal(type, 'norm_hermite')
    %%% normalized generalized Hermite polynomials
    % H_n^t(x) * t^(-n/2) * (n!)^(-.5).
    t = scale; 
    yy = grid/sqrt(t);
    A(:,1) = 1; A(:,2) = yy;
    for i = 2:order
        A(:,i+1) = yy .* A(:,i) - (i-1) * A(:,i-1);
    end
    A = A ./ sqrt(factorial(0:order));
else
    fprintf('wrong type');
end
end
