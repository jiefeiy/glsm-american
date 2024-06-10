clear, clc, close all
addpath ../utils

% price Bermudan put option under Heston model using G-LSM 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options.

% parameter ref: test No. 4 in the paper by Fang Fang andCornelis W. Oosterlee, 
%                "A Fourier-Based Valuation Method for Bermudan and Barrier
%                Options under Heston's Model"
% S0 =    [ 8          9           10         11         12];

% V_ref_1996 = [2.0000    1.1080    0.5316    0.2261    0.0907];
% V_ref_1998 = [2.0000    1.1076    0.5202    0.2138    0.0821];
% V_ref_2003 = [2.00      1.107     0.517     0.212     0.0815];
% V_ref_2009_OS = [2.00000 1.10761  0.51987   0.21353   0.08197];
% V_ref_2009_PSOR = [2.00000 1.10749 0.51985  0.21354   0.08198];
% V_ref_2011 = [2.000000   1.107621   0.520030   0.213677   0.082044]; 
% V_ref_2015 = [2.0000     1.1081     0.5204    0.2143     0.0827];

p.S0       = 10; 
p.strike   = 10;
p.v0       = 0.0625;
p.rate     = 0.1;
p.dividend = 0;
p.rho      = 0.1;
p.kappa    = 5;
p.theta    = 0.16;
p.nu       = 0.9;
p.expiration  = 0.25;
p.numTimeStep = 50;

M = 100000;
order = 20;
I = hyperbolic_cross_indices(2, order);
Nbasis = size(I,1);

%%% running parameters
num_trials = 10;
file_name = ['heston_GLSM_So' num2str(p.S0) '_M' num2str(M) '_Nb' num2str(Nbasis)...
    '_trials' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);

% warning('off');
%%% run and save 
for t = 1:num_trials
    V0_vals(t, 1) = run_heston(p, M, I);
    fprintf('run trial no.%d, price = %1.4f \n', t, V0_vals(t, 1) );
    fprintf('---------------------------------------------\n');
end

% save(['data/' file_name '.mat']);
mean(V0_vals)


function V0 = run_heston(p, M, I)
S0 = p.S0;
K = p.strike;
r = p.rate;
rho = p.rho;
nu = p.nu;
T = p.expiration;
N = p.numTimeStep;
dt = T/N;
tau = N*ones(M,1);

%%% generate paths
[Wpaths, Xpaths] = gen_paths_heston_logscale(p, M);

payoff = exp(-r*T) * max( K - S0*exp(Xpaths(:,1,N)), 0);
domain_logv = [-7, log(0.8)];
type = {'norm_hermite', 'chebyshev'};
for k = N-1:-1:1
    xstd = std(Xpaths(:,1,k), 0, 1);
    scale = {xstd^2, domain_logv};
    [A1, G] = gen_poly_basis_grad(type, I', Xpaths(:,:,k), scale); 
    dW1 = (Wpaths(:,1,k+1) - Wpaths(:,1,k));
    dW2 = (Wpaths(:,2,k+1) - Wpaths(:,2,k));
    A2 = (rho * dW1 + sqrt(1-rho^2) * dW2) .* exp(Xpaths(:,2,k)/2) .* G{1};
    A2 = A2 + (nu * dW1) .* exp(-Xpaths(:,2,k)/2) .* G{2};

    beta = (A1 + A2) \ payoff;
    CV = A1 * beta;

    EV = exp(-r*k*dt) * max( K - S0 * exp(Xpaths(:,1,k)), 0);
    idx = (CV < EV) & (EV > 0);
    tau(idx) = k;
    payoff(idx) = EV(idx);
    payoff(~idx) = CV(~idx);
end

%%% compute price at t=0
vv = zeros(M,1);
for m = 1:M
    vv(m) = exp(-r*tau(m)*dt) * max( K - S0 * exp(Xpaths(m,1,tau(m))), 0); 
end
V0 = mean(vv);
end



