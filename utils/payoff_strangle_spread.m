function valueMatrix = payoff_strangle_spread(Spaths, K, r, dt)
%Compute the payoff matrix at each timestep for all samples
%   return a M-by-N matrix
[M, ~, N] = size(Spaths);
valueMatrix = zeros(M,N);

for k = 1:N
    Smean = mean(Spaths(:,:,k), 2);
    valueMatrix(:, k) = exp(-r*k*dt)*( - max(K(1) - Smean, 0) ...
        + max(K(2) - Smean, 0) ...
        + max(Smean - K(3), 0) ...
        - max(Smean - K(4), 0) );
end

end