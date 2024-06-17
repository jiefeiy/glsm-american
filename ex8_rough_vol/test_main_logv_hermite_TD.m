clear, clc
addpath ../utils

% Define the nodes and weights for the stochastic volatility model
nodes = [0.05, 8.71708699];
weights = [0.76732702, 3.22943184];

% Define the model parameters
p.lambda = 0.3; 
p.nu = 0.3; 
p.theta = 0.02;   
p.V_0 = 0.02; 
p.T = 1; 
p.rho = -0.7; 
p.S0 = 100; 
p.r = 0.06; 
p.m = 100000;               % number of sample paths
p.N_time = 128;
p.K = 105;                  % strike price

% basis parameters
order = 10;
I = generate_index_set('TP', 2, order)'; 
Nbasis = size(I,1);

%%% running parameters
num_trials = 1;
file_name = ['rough_logv_hermite_M' num2str(p.m) '_order' num2str(order) '_N' num2str(p.N_time) ...
    '_trials' num2str(num_trials)];
V0_amer_vals = zeros(num_trials, 1);
V0_euro_vals = zeros(num_trials, 1);

%%% run and save 
for t = 1:num_trials
    [V0_amer_vals(t, 1), V0_euro_vals(t, 1)] = run_rough_logv_hermite(p, I, nodes, weights);
    fprintf('run trial no.%d, amer_price = %1.4f, euro_price = %1.4f \n', t, V0_amer_vals(t, 1), V0_euro_vals(t, 1) );
    fprintf('---------------------------------------------\n');
end

save(['data/' file_name '.mat']);
% mean(V0_amer_vals)
% mean(V0_euro_vals)


function [amer_put_price, euro_put_price] = run_rough_logv_hermite(p, I, nodes, weights)

% Define the model parameters
lambda = p.lambda; 
nu = p.nu;
theta = p.theta;
V_0 = p.V_0;
T = p.T;
rho = p.rho;
S0 = p.S0;
r = p.r;
m = p.m;               % number of sample paths
N_time = p.N_time;
K = p.K;                 % strike price
dt = T/N_time;
tau = N_time*ones(m,1);   % store stopping times

weights_l1 = sum(weights);

% Generate sample paths for the stochastic volatility model
[Xpaths, ypaths, dBW] = samples_sobol(lambda, nu, theta, V_0, T, nodes, weights, rho, S0, r, m, N_time);
Xpaths = Xpaths(:, 2:end);
Spaths = exp(Xpaths);
lnypaths = log(ypaths(:, 2:end));

% Calculate the European put option price for reference
euro_put_price = exp(-r * T) * mean( max(K - Spaths(:, end), 0) );

% Set up the basis functions for the American put option
type = 'norm_hermite';
Nbasis = size(I,1);

d = 2;
loc_grad = zeros(Nbasis, d);
for n = 1:Nbasis
    target = I(n,:) - eye(d);
    target(target<0) = 0;
    [~,loc_grad(n,:)] = ismember(target, I, 'rows');
end

% Compute the payoff matrix for the American put option
valueMatrix = payoff_put(Spaths, K, r, dt);
payoff = valueMatrix(:, end);

% Perform the backwards induction to compute the American put option price
for k = N_time-1:-1:1
    xscale = [std(Spaths(:, k), 0, 1)^2,   std(lnypaths(:, k), 0, 1)^2];
    xmean = [mean(Spaths(:, k)),   mean(lnypaths(:, k))];
    var_paths = [Spaths(:, k) lnypaths(:, k)];   % regression variables
    A1 = generate_poly_hermite_anis(type, I, var_paths - xmean, xscale); 
    % adding martingale terms 
    A = A1;
    for j = 1:2
        dW = dBW(:, j, k);
        for n = 1:Nbasis
            if I(n,j) >= 1 && j==1
                A(:,n) = A(:,n) + sqrt(1-rho^2)*Spaths(:,k) .* exp(lnypaths(:,k)/2) ...
                                .* dW .* A1(:,loc_grad(n,1)) * sqrt(I(n,1)/xscale(1));
            elseif I(n,j) >= 1 && j==2
                A(:,n) = A(:,n) + ( rho*Spaths(:,k) .*exp(lnypaths(:,k)/2) .* A1(:,loc_grad(n,1)) * sqrt(I(n,1)/xscale(1)) ...
                    + A1(:,loc_grad(n,2)) * sqrt(I(n,2)/xscale(2)) .*exp(lnypaths(:,k)/2) * nu*weights_l1  ) .* dW;
            end
        end
    end
    beta = A \ payoff;
    CV = A1*beta;               % compute continuation value
    EV = valueMatrix(:, k);      % exercise value

    idx = (CV < EV) & (EV > 0); % decide the index of points to be exercised
    tau(idx) = k;
    payoff(idx) = EV(idx);      % update the value
    payoff(~idx) = CV(~idx);

end
idx2 = sub2ind(size(valueMatrix), 1:m, tau');
amer_put_price = mean(valueMatrix(idx2));

end


%% function dependencies
function A = generate_poly_hermite_anis(type, I, grid, scale)
% Generate anisotropic polynomial Hermite basis functions
% type: 'norm_hermite' for normalized Hermite polynomials
% I: matrix of indices for the basis functions
% grid: the input data points, should be a column vector
% scale: the scaling factors for each dimension
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
for n = 1:N
    P_all = zeros(M,d);
    for j = 1:d
        P_all(:,j) = P1{j}(:, I(n,j)+1);
    end
    A(:,n) = prod(P_all,2);
end
end


function valueMatrix = payoff_put(Spaths, K, r, dt)
% Compute the discounted payoff matrix at each timestep for all samples
%   return a M-by-N matrix
[M, N_time] = size(Spaths); 
valueMatrix = zeros(M, N_time);
for k = 1:N_time
    valueMatrix(:, k) = exp(-r*k*dt)*max( K - Spaths(:,k), 0 );
end
end



