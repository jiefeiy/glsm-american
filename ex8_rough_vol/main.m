clear, clc
addpath ../utils

nodes = [0.05, 8.71708699];
weights = [0.76732702, 3.22943184];
weights_l1 = sum(weights);

lambda = 0.3; 
nu = 0.3; 
theta = 0.02;   
V_0 = 0.02; 
T = 1; 
rho = -0.7; 
S0 = 100; 
r = 0.06; 
m = 40000;
N_time = 128;
K = 105;
dt = T/N_time;
tau = N_time*ones(m,1);

[Xpaths, ypaths, dBW] = samples(lambda, nu, theta, V_0, T, nodes, weights, rho, S0, r, m, N_time);
Spaths = exp(Xpaths(:, 2:end));
lnypaths = log(ypaths(:, 2:end));
euro_put_price = exp(-r * T) * mean( max(K - Spaths(:, end), 0) )

order = 15;
I = hyperbolic_cross_indices(2, order); 
Nbasis = size(I,1);
type = 'norm_hermite';

d = 2;
loc_grad = zeros(Nbasis, d);
for n = 1:Nbasis
    target = I(n,:) - eye(d);
    target(target<0) = 0;
    [~,loc_grad(n,:)] = ismember(target, I, 'rows');
end

valueMatrix = payoff_put(Spaths, K, r, dt);
payoff = valueMatrix(:, end);
for k = N_time-1:-1:1
    xscale = [std(Spaths(:, k), 0, 1)^2,   std(lnypaths(:, k), 0, 1)^2];
    xmean = [mean(Spaths(:, k)),   mean(lnypaths(:, k))];
    var_paths = [Spaths(:, k) lnypaths(:, k)];
    A1 = generate_poly_hermite_anis(type, I, var_paths - xmean, xscale); 
    A = A1;
    for j = 1:2
        dW = dBW(:, j, k);
        for n = 1:Nbasis
            if I(n,j) >= 1 && j==1
                A(:,n) = A(:,n) + sqrt(1-rho^2) *exp(lnypaths(:,k)/2) .* dW .* A1(:,loc_grad(n,1)) * sqrt(I(n,1)/xscale(1));
            elseif I(n,j) >= 1 && j==2
                A(:,n) = A(:,n) + ( rho*exp(lnypaths(:,k)/2) .* A1(:,loc_grad(n,1)) * sqrt(I(n,1)/xscale(1)) ...
                    + A1(:,loc_grad(n,2)) * sqrt(I(n,2)/xscale(2)) .*exp(lnypaths(:,k)/2) * nu*weights_l1  ) .* dW;
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
    k
end
idx = sub2ind(size(valueMatrix), 1:m, tau');
amer_put_price = mean(valueMatrix(idx))




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

function valueMatrix = payoff_put(Spaths, K, r, dt)
%Compute the discounted payoff matrix at each timestep for all samples
%   return a M-by-N matrix
[M, N] = size(Spaths); 
valueMatrix = zeros(M, N);
for k = 1:N
    valueMatrix(:, k) = exp(-r*k*dt)*max( K - Spaths(:,k), 0 );
end
end


function [Xpaths, ypaths, dBW] = samples(lambda, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time)
dt = T / N_time;
N = length(nodes);
V_init = V_0 ./ nodes ./ sum(weights ./ nodes);
A = eye(N) + diag(nodes) * dt + lambda * weights' * dt;
A_inv = inv(A);
b = theta * dt + (nodes .* V_init) * dt;

Xpaths = zeros(m, N_time+1); 
ypaths = zeros(m, N_time+1);
dBW = sqrt(dt) * randn(m, 2, N_time);

current_V_comp = repmat(V_init, m, 1);
current_log_S = log(S_0) * ones(m, 1);
Xpaths(:, 1) = log(S_0);
ypaths(:, 1) = V_init * weights';

for i = 1:N_time
    sq_V = sqrt(max(current_V_comp * weights', 0));
    current_log_S = current_log_S + r * dt + sq_V .* (rho * dBW(:, 2, i) + sqrt(1 - rho^2) * dBW(:, 1, i)) ...
        - 0.5 * sq_V.^2 * dt;
    current_V_comp = (current_V_comp + nu * (sq_V .* dBW(:, 2, i)) + b) * A_inv;

    Xpaths(:, i+1) = current_log_S;
%     ypaths(:, i+1) = max(current_V_comp * weights', 0);
    ypaths(:, i+1) = abs(current_V_comp * weights');
end

end

