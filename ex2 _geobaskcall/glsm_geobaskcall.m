clear, clc, close all
% price Bermudan geometric basket put option using G-LSM 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options

addpath ../utils
poolobj = parpool;
fprintf('Number of workers: %g\n', poolobj.NumWorkers);

%%% set parameters
p.strike = 100; p.rate = 0; p.dividend = 0.02;
p.expiration = 2;
p.dim = 2;                                          % asset number
p.S0 = 100*ones(p.dim,1);
p.volatility = diag(ones(p.dim,1))*0.25;
p.correlation = 0.25*eye(p.dim) + 0.75*ones(p.dim);
p.numTimeStep = 50;
p.callput = 'call';

M = 100000;
order = 10;
I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

%%% running parameters
num_trials = 2; 
file_name = ['a1_geobaskcall_hermite_d' num2str(p.dim) '_M' num2str(M)...
    '_order' num2str(order) '_Nb' num2str(Nbasis) '_trials' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);
delta0_vals = zeros(num_trials, p.dim);

%%% run and save
for t = 1:num_trials
    [V0_vals(t, 1), delta0_vals(t,:)] = run_glsm_geobaskcall(p, M, order);
    fprintf('run trial no.%d, price = %1.4f \n',  t, V0_vals(t, 1) );
    disp(['delta: ' num2str(delta0_vals(t,:))]);
    fprintf('---------------------------------------------\n');
end

% save(['data/' file_name '.mat']);
mean(V0_vals)
mean(delta0_vals, 1)

delete(gcp('nocreate'))

function [V0, delta0] = run_glsm_geobaskcall(p, M, order)
type = 'norm_hermite';
K = p.strike;
r = p.rate;
T = p.expiration;
d = p.dim;
S0 = p.S0;
sig = p.volatility;
P = p.correlation;
N = p.numTimeStep;
dt = T/N;
tau = N*ones(M,1);

I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

[Wpaths,Spaths] = gen_paths_multi_bs(p, M);
valueMatrix = payoff_geo(Spaths, K, r, dt, p.callput);
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
    A1 = generate_poly_hermite(type, I, Wpaths(:,:,k), scale); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    beta = cgs(A'*A/M, A'*payoff/M);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CV = A1*beta;
    clear A1 A 
    EV = valueMatrix(:,k);

    idx = (CV < EV) & (EV > 0);
    tau(idx) = k;
    payoff(idx) = EV(idx); 
    payoff(~idx) = CV(~idx);
end
%%% compute price at t=0
idx = sub2ind(size(valueMatrix), 1:M, tau');
V0 = mean(valueMatrix(idx));

%%% compute delta at t=0 %%%%
A1 = generate_poly_hermite(type, I, zeros(1,d), scale); 
A = repmat(A1, M, 1);
for j = 1:d
    dW = Wpaths(:,j,1);
    for n = 1:Nbasis
        if I(n,j) >= 1 
            A(:,n) = A(:,n) + dW .* A1(:,loc_grad(n,j)) * sqrt(I(n,j)/scale);
        end
    end
end
beta = cgs(A'*A/M, A'*payoff/M);
Z0 = zeros(d,1);
for j = 1:d
    G = zeros(1, Nbasis);
    for n = 1:Nbasis
        if I(n,j) >= 1
            G(n) = A1(:,loc_grad(n,j)) * sqrt(I(n,j)/scale);
        end
    end
    Z0(j) = G * beta;
end
[Q, Lambda] = eig(sig*P*sig');
[diag_ele, ind] = sort(diag(Lambda),'descend');
Q = Q(:, ind); Lambda = diag(diag_ele);          
dw_ds = Q ./ sqrt(diag(Lambda))' ./ S0;
delta0 = dw_ds * Z0;
delta0 = delta0';
end



