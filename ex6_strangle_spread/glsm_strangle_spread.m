clear, clc, close all
% price a strangle spread basket option
% 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options
% [2] Kohler, M., Krzyżak, A., & Todorovic, N. (2010). Pricing of 
% High‐Dimensional American Options by Neural Networks. Mathematical Finance

addpath ../utils
%%% set parameters
p.strike = [75 90 110 125];
p.rate = 0.05; 
p.dividend = 0;
p.expiration = 1;
p.dim = 5;                                   % asset number
p.S0 = 100*ones(p.dim,1);
p.volatility = [0.3024   0.1354   0.0722   0.1367   0.1641;
    0.1354   0.2270   0.0613   0.1264   0.1610;
    0.0722   0.0613   0.0717   0.0884   0.0699;
    0.1367   0.1264   0.0884   0.2937   0.1394;
    0.1641   0.1610   0.0699   0.1394   0.2535];
p.correlation = eye(p.dim);
p.numTimeStep = 48;                     

M = 200000;
order = 15;                                  % polynomials up to the order 
I = hyperbolic_cross_indices(p.dim, order);   % generate hyperbolic cross index set
Nbasis = size(I,1);

%%% running parameters
num_trials = 10;
file_name = ['strangle_spread_basket_M' num2str(M) '_order' num2str(order) ...
    '_Nb' num2str(Nbasis) '_trails' num2str(num_trials)];
V0_vals = zeros(num_trials, 1);
time_vals = zeros(num_trials, 5);

%%% run and save
for t = 1:num_trials
    [V0_vals(t, 1), time_vals(t, :)] = run_glsm_strangle_spread(p, M, order);
    fprintf('run trial no.%d, price = %1.4f \n',  t, V0_vals(t, 1) );
    disp(['Times: ' num2str(time_vals(t,:))]);
    fprintf('---------------------------------------------\n');
end

save(['data/' file_name '.mat']);
mean(V0_vals)
std(V0_vals)
mean(time_vals, 1)

function [V0, time] = run_glsm_strangle_spread(p, M, order)
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
valueMatrix = payoff_strangle_spread(Spaths, K, r, dt);
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
    beta = A \ payoff;
%     beta = cgs(A'*A/M, A'*payoff/M);
    tend_sub3 = toc(t_sub3);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t_sub4 = tic;
    CV = A1*beta;               % compute continuation value
    EV = valueMatrix(:,k);      % exercise value
     
    idx = (CV < EV) & (EV > 0); % decide the index of points to be exercised
    tau(idx) = k;
    payoff(idx) = EV(idx);      % update the value
    payoff(~idx) = CV(~idx);
    tend_sub4 = toc(t_sub4);
end
%%% compute price at t=0
idx = sub2ind(size(valueMatrix), 1:M, tau');
V0 = mean(valueMatrix(idx));
tTotal = toc(tStart);

time = [tend_sub1, tend_sub2, tend_sub3, tend_sub4, tTotal];
end

