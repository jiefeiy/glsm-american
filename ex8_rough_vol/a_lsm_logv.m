clear, clc
addpath ../utils

% Define the nodes and weights for the stochastic volatility model
nodes = [0.05, 8.71708699];
weights = [0.76732702, 3.22943184];
weights_l1 = sum(weights);

% Define the model parameters
lambda = 0.3; 
nu = 0.3; 
theta = 0.02;   
V_0 = 0.02; 
T = 1; 
rho = -0.7; 
S0 = 100; 
r = 0.06; 
m = 100000;               % number of sample paths
N_time = 128;
K = 105;                  % strike price
dt = T/N_time;
tau = N_time*ones(m,1);   % store stopping times

% Generate sample paths for the stochastic volatility model
[Xpaths, ypaths, dBW] = samples_sobol(lambda, nu, theta, V_0, T, nodes, weights, rho, S0, r, m, N_time);
Xpaths = Xpaths(:, 2:end);
Spaths = exp(Xpaths);
lnypaths = log(ypaths(:, 2:end));

% Calculate the European put option price for reference
euro_put_price = exp(-r * T) * mean( max(K - Spaths(:, end), 0) )

% Set up the basis functions for the American put option
order = 10;
% I = hyperbolic_cross_indices(2, order); 
I = generate_index_set('TP', 2, order)'; 
Nbasis = size(I,1);
type = 'norm_hermite';

% Compute the payoff matrix for the American put option
valueMatrix = payoff_put(Spaths, K, r, dt);
payoff = valueMatrix(:, end);

% Perform the backwards induction to compute the American put option price
for k = N_time-1:-1:1
    xscale = [std(Spaths(:, k), 0, 1)^2,   std(lnypaths(:, k), 0, 1)^2];
    xmean = [mean(Spaths(:, k)),   mean(lnypaths(:, k))];
    var_paths = [Spaths(:, k) lnypaths(:, k)];   % regression variables
    A1 = generate_poly_hermite_anis(type, I, var_paths - xmean, xscale); 
    beta = A1 \ payoff;
    CV = A1*beta;               % compute continuation value
    EV = valueMatrix(:, k);      % exercise value

    idx = (CV < EV) & (EV > 0); % decide the index of points to be exercised
    tau(idx) = k;
    payoff(idx) = EV(idx);      % update the value
    k

    if k == (N_time/2)
        hold on; plot(Spaths(~idx, k), lnypaths(~idx, k), '.', 'Color', 'r');
        plot(Spaths(idx, k), lnypaths(idx, k), '.', 'Color', 'b');
        % plot(K*ones(1,10), linspace(min(lnypaths(:,k)), max(lnypaths(:,k)), 10), '-k' );
        xlim([0,160]); ylim([-16,0]);
        xlabel('S_t'); ylabel('ln(y_t)'); 
        legend('continue', 'exercise');
    end
end
% idx2 = sub2ind(size(valueMatrix), 1:m, tau');
% amer_put_price = mean(valueMatrix(idx2))
amer_put_price = mean(payoff)


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


