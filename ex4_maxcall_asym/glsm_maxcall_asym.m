clear, clc, close all
% price Bermudan geometric basket put option using G-LSM 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options

addpath ../utils
poolobj = parpool;
fprintf('Number of workers: %g\n', poolobj.NumWorkers);

%%% set parameters
p.strike = 100; p.rate = 0.05; p.dividend = 0.1;
p.expiration = 3;
p.dim = 2;                                          % asset number
S0_i = 100;
p.S0 = S0_i*ones(p.dim,1);
if p.dim <= 5
    p.volatility = diag(0.08 + 0.32*(0:p.dim-1)/(p.dim-1));
else 
    p.volatility = diag(0.1 + (1:p.dim)/(2*p.dim));
end
p.correlation = eye(p.dim);
p.numTimeStep = 9;
 
M = 100000;
order = 10;
I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

%%% running parameters
num_trials = 10; 
file_name = ['a1_maxcall_asym_d' num2str(p.dim) '_S' num2str(S0_i) '_M' num2str(M)...
    '_order' num2str(order) '_Nb' num2str(Nbasis) '_trials' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);
time_vals = zeros(num_trials, 5);

%%% run and save
for t = 1:num_trials
    [V0_vals(t, 1), time_vals(t,:)] = run_a1_maxcall(p, M, order);
    fprintf('run trial no.%d, price = %1.4f \n',  t, V0_vals(t, 1) );
    disp(['Times: ' num2str(time_vals(t,:))]);
    fprintf('---------------------------------------------\n');
end

% save(['data_asym/' file_name '.mat']);
mean(V0_vals)
mean(time_vals, 1)

delete(gcp('nocreate'))

function [V0, time] = run_a1_maxcall(p, M, order)
type = 'norm_hermite';
K = p.strike;
r = p.rate;
T = p.expiration;
d = p.dim;
N = p.numTimeStep;
dt = T/N;
tau = N*ones(M,1);

I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

[Wpaths,Spaths] = gen_paths_multi_bs(p, M);
valueMatrix = payoff_maxcall(Spaths, K, r, dt);
loc_grad = zeros(Nbasis, d);
for n = 1:Nbasis
    target = I(n,:) - eye(d);
    target(target<0) = 0;
    [~,loc_grad(n,:)] = ismember(target, I, 'rows');
end

% Dynamic programming
tStart = tic;
payoff = valueMatrix(:,N);
for k = N-1:-1:1
    scale = k*dt;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_sub1 = tic;
    A1 = generate_poly_hermite(type, I, Wpaths(:,:,k), scale); 
    tend_sub1 = toc(t_sub1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_sub2 = tic;
    A = A1;
    for j = 1:d
        dW = (Wpaths(:,j,k+1) - Wpaths(:,j,k));
        for n = 1:Nbasis
            if I(n,j) >= 1
                A(:,n) = A(:,n) + dW .* A1(:,loc_grad(n,j)) * sqrt(I(n,j)/scale);
            end
        end
    end
    tend_sub2 = toc(t_sub2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_sub3 = tic;
    beta = cgs(A'*A/M, A'*payoff/M);
    tend_sub3 = toc(t_sub3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_sub4 = tic;
    CV = A1*beta;
    clear A1 A 
    EV = valueMatrix(:,k);
    
    idx = (CV < EV) & (EV > 0);
    tau(idx) = k;
    payoff(idx) = EV(idx); 
    payoff(~idx) = CV(~idx);
    tend_sub4 = toc(t_sub4);
end
%%% compute price at t=0
idx = sub2ind(size(valueMatrix), 1:M, tau');
V0 = mean(valueMatrix(idx));
tTotal = toc(tStart);

time = [tend_sub1, tend_sub2, tend_sub3, tend_sub4, tTotal];
end