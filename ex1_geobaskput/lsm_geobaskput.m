clear, clc, close all
% price Bermudan geometric basket put option using LSM 
% Reference: 
% [1] [Longstaff and Schwartz, 2001] Longstaff, F. and Schwartz, E. (2001). Valuing American
% options by simulation: a simple least-squares approach. The review of financial studies,
% 14(1):113–147

addpath ../utils
poolobj = parpool;
fprintf('Number of workers: %g\n', poolobj.NumWorkers);

%%% set parameters
p.strike = 100; p.rate = 0.03; p.dividend = 0;
p.expiration = 0.25;
p.dim = 2;                                          % asset number
p.S0 = 100*ones(p.dim,1);
p.volatility = diag(ones(p.dim,1))*0.2;
p.correlation = 0.5*eye(p.dim) + 0.5*ones(p.dim);
p.numTimeStep = 50;
p.callput = 'put';

M = 100000;
order = 10;
I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

%%% running parameters
num_trials = 1; 
file_name = ['geobaskput_LSM_d' num2str(p.dim) '_M' num2str(M)...
    '_order' num2str(order) '_Nb' num2str(Nbasis) '_trials' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);

%%% run and save
for t = 1:num_trials
    V0_vals(t, 1) = run_geobaskput_lsm(p, M, order);
    fprintf('run trial no.%d, price = %1.4f \n',  t, V0_vals(t, 1) );
    fprintf('---------------------------------------------\n');
end

% save(['data/' file_name '.mat']);
mean(V0_vals)

function V0 = run_geobaskput_lsm(p, M, order)
type = 'norm_hermite';
K = p.strike;
r = p.rate;
T = p.expiration;
N = p.numTimeStep;
dt = T/N;

I = hyperbolic_cross_indices(p.dim, order);
[Wpaths,Spaths] = gen_paths_multi_bs(p, M);
valueMatrix = payoff_geo(Spaths, K, r, dt, p.callput);

% Dynamic programming
payoff = valueMatrix(:,N);
for k = N-1:-1:1
    scale = k*dt;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    A = generate_poly_hermite(type, I, Wpaths(:,:,k), scale); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    beta = cgs(A'*A/M, A'*payoff/M);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CV = A*beta;
    clear A 
    EV = valueMatrix(:,k);

    idx = (CV < EV) & (EV > 0);
    payoff(idx) = EV(idx); 
end
%%% compute price at t=0
V0 = mean(payoff);
end
