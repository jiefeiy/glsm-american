function valueMatrix = payoff_maxcall(Spaths, K, r, dt)
%Compute the payoff matrix at each timestep for all samples
%   return a M-by-N matrix
[M, ~, N] = size(Spaths);
valueMatrix = zeros(M,N);

for k = 1:N
    valueMatrix(:, k) = exp(-r*k*dt)*max( max(Spaths(:,:,k), [], 2) - K, 0 );
end

end

