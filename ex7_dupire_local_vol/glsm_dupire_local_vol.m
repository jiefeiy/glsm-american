clear, clc, close all
% price a arithmetic basket put under Dupire's local volatility
% 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options
% [2] Becker, S., Cheridito, P., Jentzen, A., & Welti, T. (2021). 
% Solving high-dimensional optimal stopping problems using deep learning. 

addpath ../utils
%%% set parameters
p.strike = 100;
p.rate = 0.05; 
p.dividend = 0.1;
p.expiration = 1;
p.dim = 5;                                   % asset number
p.S0 = 100;
p.correlation = eye(p.dim);
p.numTimeStep = 50;                     

M = 200000;
order = 16;                                  % polynomials up to the order 
I = hyperbolic_cross_indices(p.dim, order);   % generate hyperbolic cross index set
Nbasis = size(I,1);

%%% running parameters
num_trials = 10;
file_name = ['dupire_baskput_di' num2str(p.dividend) '_N' num2str(p.numTimeStep) '_M' num2str(M) '_order' num2str(order) ...
    '_Nb' num2str(Nbasis) '_trails' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);

%%% run and save 
for t = 1:num_trials
    V0_vals(t, 1) = run_glsm_dupire_baskput(p, M, order);
    fprintf('run trail no.%d, price = %1.4f \n', t, V0_vals(t,1) );
    fprintf('---------------------------------------------\n');
end

save(['data/' file_name '.mat']);
mean(V0_vals)
std(V0_vals)


function V0 = run_glsm_dupire_baskput(p, M, order)
type = 'norm_hermite';
S0 = p.S0;
K = p.strike;
r = p.rate;
T = p.expiration;
d = p.dim;
N = p.numTimeStep;
dt = T/N;
tau = N*ones(M,1);
vol_fun = @(t, x) 0.6 * exp(-0.05*sqrt(t)) .* ( ...
    1.2 - exp(-0.1*t - 0.001*S0^2*(exp(r*t + x) - 1).^2) ...
    );

I = hyperbolic_cross_indices(p.dim, order); 
Nbasis = size(I,1);

[Wpaths,Xpaths] = gen_paths_dupire(p, M);
valueMatrix = payoff_arith_basket(Xpaths, K, r, dt, S0);
loc_grad = zeros(Nbasis, d);
for n = 1:Nbasis
    target = I(n,:) - eye(d);
    target(target<0) = 0;
    [~,loc_grad(n,:)] = ismember(target, I, 'rows');
end

payoff = valueMatrix(:,N);
for k = N-1:-1:1
    scale = std(Xpaths(:, :, k), 0, 1).^2;
    A1 = generate_poly_hermite_anis(type, I, Xpaths(:,:,k), scale); 
    A = A1;
    for j = 1:d
        dW = (Wpaths(:,j,k+1) - Wpaths(:,j,k));
        vol = vol_fun(k*dt, Xpaths(:, j, k));
        for n = 1:Nbasis
            if I(n,j) >= 1
                A(:,n) = A(:,n) + vol .* dW .* A1(:,loc_grad(n,j)) * sqrt(I(n,j)/scale(j));
            end
        end
    end
    beta = A \ payoff;
    CV = A1*beta;               % compute continuation value
    EV = valueMatrix(:,k);      % exercise value

    idx = (CV < EV) & (EV > 0); % decide the index of points to be exercised
    tau(idx) = k;
    payoff(idx) = EV(idx);      % update the value
    payoff(~idx) = CV(~idx);
end
idx = sub2ind(size(valueMatrix), 1:M, tau');
V0 = mean(valueMatrix(idx));
end


function A = generate_poly_hermite_anis(type, I, grid, scale)
% grid should be a column vector
[N,d] = size(I);       % get N (number of matrix columns) and d (dimension)
M = size(grid,1);      % get m (number of matrix rows)
A = zeros(M,N);        % initialize A
order = max(I(:));     % find maximum polynomial degree
P1 = cell(1,d);        % store 1d basis
for j = 1:d
    yy = grid(:,j);
    P1{j} = generate_poly_basis_1d(type, order, yy, scale(j));   % M-by-(order+1) matrix
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


function valueMatrix = payoff_arith_basket(Xpaths, K, r, dt, S0)
%Compute the payoff matrix at each timestep for all samples
%   return a M-by-N matrix
[M, ~, N] = size(Xpaths);
valueMatrix = zeros(M,N);
for k = 1:N
    valueMatrix(:, k) = exp(-r*k*dt)*max( K - S0*mean(exp(Xpaths(:,:,k)), 2), 0 );
end
end


function [W,X] = gen_paths_dupire(p, M)
r = p.rate;
di = p.dividend;
T = p.expiration;
d = p.dim;
S0 = p.S0;
N = p.numTimeStep;
dt = T/N; 
vol_fun = @(t, x) 0.6 * exp(-0.05*sqrt(t)) .* ( ...
    1.2 - exp( -0.1*t - 0.001*S0^2*(exp(r*t + x) - 1).^2 ) ...
    );
% generate paths
W = zeros(M, d, N); X = W; 
W(:, :, 1) = sqrt(dt)*randn(M,d);
vol = vol_fun(0, 0);
X(:, :, 1) = (r - di - .5*vol^2)*dt + vol * W(:, :, 1);

for k = 1:N-1 
    dW = sqrt(dt)*randn(M,d);
    W(:, :, k+1) = W(:,:,k) + dW;
    
    vol = vol_fun(k*dt, X(:, :, k));
    X(:, :, k+1) = X(:, :, k) + (r - di - .5*vol.^2)*dt + vol .* dW;
end
end

