function valueMatrix = payoff_geo(Spaths, K, r, dt, callput)
%Compute the payoff matrix at each timestep for all samples
%   return a M-by-N matrix
[M, ~, N] = size(Spaths);
valueMatrix = zeros(M,N);

if isequal(callput,'put')
    for k = 1:N
        valueMatrix(:,k) = exp(-r*k*dt)*max(K - geomean( Spaths(:,:,k) , 2), 0); 
    end
elseif isequal(callput,'call')
    for k = 1:N
        valueMatrix(:,k) = exp(-r*k*dt)*max(geomean( Spaths(:,:,k), 2) - K, 0); 
    end
end

end

