function [Xpaths, ypaths, dBW] = samples_sobol(lambda, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time)
% Generate sample paths for the stochastic volatility model using Euler–Maruyama method
% using Sobol sequence for rng
dt = T / N_time;
N = length(nodes);
V_init = V_0 ./ nodes ./ sum(weights ./ nodes);
A = eye(N) + diag(nodes) * dt + lambda * weights' * dt;
A_inv = inv(A);
b = theta * dt + (nodes .* V_init) * dt;

Xpaths = zeros(m, N_time+1); 
ypaths = zeros(m, N_time+1);
p = sobolset(2*N_time,'Skip',1e3,'Leap',1e2);
random_gaussian = norminv( p(1:m, :) );
dBW = sqrt(dt) * reshape(random_gaussian, [m, 2, N_time]);
% dBW = sqrt(dt) * randn(m, 2, N_time);          % store [W^\perp, W]

current_V_comp = repmat(V_init, m, 1);
current_log_S = log(S_0) * ones(m, 1);
Xpaths(:, 1) = log(S_0);
ypaths(:, 1) = V_init * weights';

for i = 1:N_time
    sq_V = sqrt(max(current_V_comp * weights', 0));
    current_log_S = current_log_S + r * dt + sq_V .* (rho * dBW(:, 2, i) + sqrt(1 - rho^2) * dBW(:, 1, i)) ...
        - 0.5 * sq_V.^2 * dt;
    current_V_comp = (current_V_comp + nu * (sq_V .* dBW(:, 2, i)) + b) * A_inv;

    Xpaths(:, i+1) = current_log_S;
%     ypaths(:, i+1) = max(current_V_comp * weights', 0);    % may have zero volatility
    ypaths(:, i+1) = abs(current_V_comp * weights');         % avoid negative and zero volatility
end

end