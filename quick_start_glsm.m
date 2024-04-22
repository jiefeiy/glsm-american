clear, clc, close all
% price Bermudan max-call option using G-LSM 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options

addpath ../utils
%%% set parameters
p.strike = 100; p.rate = 0.05; p.dividend = 0.1;
p.expiration = 3;
p.dim = 5;                                   % asset number
p.S0 = 100*ones(p.dim,1);
p.volatility = diag(ones(p.dim,1))*0.2;
p.correlation = eye(p.dim);
p.numTimeStep = 9;                           % N = 9

M = 200000;
order = 10;                                  % polynomials up to the order 
type = 'norm_hermite';
K = p.strike;
r = p.rate;
T = p.expiration;
d = p.dim;
N = p.numTimeStep;
dt = T/N;
tau = N*ones(M,1);

I = hyperbolic_cross_indices(p.dim, order);   % generate hyperbolic cross index set
Nbasis = size(I,1);

[Wpaths,Spaths] = gen_paths_multi_bs(p, M);
valueMatrix = payoff_maxcall(Spaths, K, r, dt);

% determine the location of gradient basis for assembling matrix A
loc_grad = zeros(Nbasis, d);
for n = 1:Nbasis
    target = I(n,:) - eye(d);
    target(target<0) = 0;
    [~,loc_grad(n,:)] = ismember(target, I, 'rows');
end

% Dynamic programming
payoff = valueMatrix(:,N);
for k = N-1:-1:1
    scale = k*dt;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generate basis matrix
    A1 = generate_poly_hermite_dir(type, I, Wpaths(:,:,k), scale); 
    % use utils/generate_poly_hermite_par.m instead for efficiency in large scale
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % assemble coefficient matrix of linear system 
    A = A1;
    for j = 1:d
        dW = (Wpaths(:,j,k+1) - Wpaths(:,j,k));
        for n = 1:Nbasis
            if I(n,j) >= 1
                A(:,n) = A(:,n) + dW .* A1(:,loc_grad(n,j)) * sqrt(I(n,j)/scale);
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % solving linear system 
    beta = A \ payoff;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CV = A1*beta;               % compute continuation value
    EV = valueMatrix(:,k);      % exercise value
     
    idx = (CV < EV) & (EV > 0); % decide the index of points to be exercised
    tau(idx) = k;
    payoff(idx) = EV(idx);      % update the value
    payoff(~idx) = CV(~idx);
    disp(['computing time step k=' num2str(k)]);
end
%%% compute price at t=0
idx = sub2ind(size(valueMatrix), 1:M, tau');
V0 = mean(valueMatrix(idx));
fprintf('The option price is %1.4f. \n', V0);

function A = generate_poly_hermite_dir(type, I, grid, scale)
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
for n = 1:N
    P_all = zeros(M,d);
    for j = 1:d
        P_all(:,j) = P1{j}(:, I(n,j)+1);
    end
    A(:,n) = prod(P_all,2);
end
end