clear, clc, close all
% price Bermudan geometric basket put option using G-LSM 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options

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
num_trials = 10; 
file_name = ['geobaskput_GLSM_d' num2str(p.dim) '_M' num2str(M)...
    '_order' num2str(order) '_Nb' num2str(Nbasis) '_trials' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);

%%% run and save
for t = 1:num_trials
    V0_vals(t, 1) = run_geobaskput(p, M, order);
    fprintf('run trial no.%d, price = %1.4f \n',  t, V0_vals(t, 1) );
    fprintf('---------------------------------------------\n');
end

% save(['data/' file_name '.mat']);
mean(V0_vals)

delete(gcp('nocreate'))

function V0 = run_geobaskput(p, M, order)
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

end