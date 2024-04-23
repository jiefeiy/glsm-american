clear, clc, close all
% price Bermudan geometric basket put option using G-LSM 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options

addpath ../utils
%%% set parameters
p.strike = 100; p.rate = 0.05; p.dividend = 0.1;
p.expiration = 3;
p.dim = 2;                                          % asset number
p.S0 = 100*ones(p.dim,1);
p.volatility = diag(ones(p.dim,1))*0.2;
p.correlation = eye(p.dim);
p.numTimeStep = 9;
k_plot = 4; % plot classification at t_k

M = 100000;
order = 20;
I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);
disp(['The number of basis function is ' num2str(Nbasis) '.']);

type = 'norm_hermite';
K = p.strike;
r = p.rate;
di = p.dividend;
T = p.expiration;
d = p.dim;
S0 = p.S0;
sig = p.volatility;
P = p.correlation;
N = p.numTimeStep;
dt = T/N;
tau = N*ones(M,1);

%%% determine the gradient basis location
loc_grad = zeros(Nbasis, d);
for n = 1:Nbasis
    target = I(n,:) - eye(d);
    target(target<0) = 0;
    [~,loc_grad(n,:)] = ismember(target, I, 'rows');
end

% generate paths
[Wpaths,Spaths] = gen_paths_multi_bs(p, M);
valueMatrix = payoff_maxcall(Spaths, K, r, dt);
% Dynamic programming
payoff = exp(-r*T)*max( max(Spaths(:,:,N), [], 2) - K, 0 ); 
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
    beta = A\payoff;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CV = A1*beta;
    clear A1 A 
    EV = valueMatrix(:,k); 
    idx = (CV < EV) & (EV > 0);
    tau(idx) = k;
    payoff(idx) = EV(idx); 
    payoff(~idx) = CV(~idx);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp(['computing k = ' num2str(k)])

    if k==k_plot
        %%% plot classification       
        scatter(Spaths(~idx,1,k) , Spaths(~idx,2,k), 3, 'r', 'filled'); hold on;    
        scatter(Spaths(idx,1,k) , Spaths(idx,2,k), 3, 'b', 'filled'); hold on; % labeled as exercise
    end

end
%%% compute price at t=0
idx = sub2ind(size(valueMatrix), 1:M, tau');
V0 = mean(valueMatrix(idx));

% title(['At time t_' num2str(k_plot)]);
xlabel('S_t^1'); ylabel('S_t^2');
xlim([0, 250]); ylim([0, 250]);
legend('continue', 'exercise');

ax = gca;
ax.FontSize = 16;
% exportgraphics(ax,['glsm_maxcall_t_' num2str(k_plot) '.eps'],'Resolution',300)
