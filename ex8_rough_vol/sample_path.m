clear, clc

nodes = [0.05, 8.71708699];
weights = [0.76732702, 3.22943184];

lambda = 0.3; 
nu = 0.3; 
theta = 0.02;   
V_0 = 0.02; 
T = 1; 
rho = -0.7; 
S_0 = 100; 
r = 0.06; 
m = 100000;
N_time = 100;

[result, dBW] = samples(lambda, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time);
Spaths = exp(squeeze(result(:,1,:)));
ypaths = squeeze(result(:,2,:));
euro_put_price = exp(-r * T) * mean( max(105 - Spaths(:, end), 0) )


time = (0:N_time)*T/N_time;
figure(1);
plot(time, Spaths(1:50, :));
xlabel('time'); ylabel('S_t');
ax = gca;
ax.FontSize = 16;
exportgraphics(ax,'rough_price_H0.1.eps','Resolution',300)

figure(2);
plot(time, log(ypaths(1:50, :)));
xlabel('time'); ylabel('y_t');
ax = gca;
ax.FontSize = 16;
exportgraphics(ax,'rough_y_H0.1.eps','Resolution',300)



function [result, dBW] = samples(lambda, nu, theta, V_0, T, nodes, weights, rho, S_0, r, m, N_time)
dt = T / N_time;
N = length(nodes);
V_init = V_0 ./ nodes ./ sum(weights ./ nodes);

A = eye(N) + diag(nodes) * dt + lambda * weights' * dt;
A_inv = inv(A);
b = theta * dt + (nodes .* V_init) * dt;

result = zeros(m, N+2, N_time+1);
dBW = sqrt(dt) * randn(m, 2, N_time);

current_V_comp = repmat(V_init, m, 1);
current_log_S = log(S_0) * ones(m, 1);
result(:, 1, 1) = current_log_S;
result(:, 3:end, 1) = repmat(V_init, m, 1);
result(:, 2, 1) = V_init * weights';

for i = 1:N_time
    sq_V = sqrt(max(current_V_comp * weights', 0));
    current_log_S = current_log_S + r * dt + sq_V .* (rho * dBW(:, 2, i) + sqrt(1 - rho^2) * dBW(:, 1, i)) ...
        - 0.5 * sq_V.^2 * dt;
    current_V_comp = (current_V_comp + nu * (sq_V .* dBW(:, 2, i)) + b) * A_inv;

    result(:, 1, i+1) = current_log_S;
    result(:, 3:end, i+1) = current_V_comp;
    result(:, 2, i+1) = max(current_V_comp * weights', 0);
end

end

