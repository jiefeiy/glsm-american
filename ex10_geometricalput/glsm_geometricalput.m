% clear, clc, close all
% price Bermudan geometric basket put option using G-LSM 
% Reference:
% [1] Hur´e, H. Pham, and X. Warin (2020). Deep backward schemes for 
% high-dimensional nonlinear PDEs
% [2] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options

% addpath ../utils

%%% set parameters
p.strike = 1; p.rate = 0.05; p.dividend = 0;
p.expiration = 1;
p.dim = 40;                                          % asset number
p.S0 = ones(p.dim,1);
p.volatility = eye(p.dim)*0.2;
p.correlation = eye(p.dim);
% p.numTimeStep = 10;
p.callput = 'put';

M = 10000;
order = 6;
I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

%%% running parameters
num_trials = 10; 
file_name = ['geometrical_GLSM_d' num2str(p.dim) '_N' num2str(p.numTimeStep) '_M' num2str(M)...
    '_order' num2str(order) '_Nb' num2str(Nbasis) '_trials' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);

%%% run and save
for t = 1:num_trials
    V0_vals(t, 1) = run_geometrical(p, M, order);
    fprintf('run trial no.%d, price = %1.4f \n',  t, V0_vals(t, 1) );
    fprintf('---------------------------------------------\n');
end

save(['data/' file_name '.mat']);
mean(V0_vals)


function V0 = run_geometrical(p, M, order)
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
valueMatrix = payoff_geometrical(Spaths, K, r, dt, p.callput);
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
    % beta = cgs(A'*A/M, A'*payoff/M);
    beta = A \ payoff;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CV = A1*beta;
    clear A1 A 
    EV = valueMatrix(:,k);
    
    idx = (CV < EV) & (EV > 0);
    tau(idx) = k;
    payoff(idx) = EV(idx); 
    payoff(~idx) = CV(~idx);
    % disp(k)
end
mean(payoff)
%%% compute price at t=0
idx = sub2ind(size(valueMatrix), 1:M, tau');
V0 = mean(valueMatrix(idx));
end


function valueMatrix = payoff_geometrical(Spaths, K, r, dt, callput)
%Compute the payoff matrix at each timestep for all samples
%   return a M-by-N matrix
[M, ~, N] = size(Spaths);
valueMatrix = zeros(M,N);

for k = 1:N
    valueMatrix(:,k) = exp(-r*k*dt)*max(K - prod( Spaths(:,:,k) , 2), 0); 
end
end
