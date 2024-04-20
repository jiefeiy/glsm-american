function [W,S] = gen_paths_multi_bs(p, M)
r = p.rate;
di = p.dividend;
T = p.expiration;
d = p.dim;
S0 = p.S0;
vol = p.volatility;
P = p.correlation;
N = p.numTimeStep;
dt = T/N;

% compute transformation
[Q, Lambda] = eig(vol*P*vol');
[diag_ele, ind] = sort(diag(Lambda),'descend');
Q = Q(:, ind); Lambda = diag(diag_ele);                  % eigen pairs of \Sigma*P*\Sigma^\top
mu = Q'*(r - di -.5*vol^2*ones(d,1));                          % sigma = sqrt(diag(Lambda))
SIG = sqrt(diag(Lambda))';

% generate paths
W = zeros(M, d, N); S = W; 
for k = 0:N-1 
    if k == 0
        W(:, :, k+1) = sqrt(dt)*randn(M,d);             
    else
        W(:, :, k+1) = W(:,:,k) + sqrt(dt)*randn(M,d);
    end
    
    logprice = mu'*(k+1)*dt + SIG.*W(:,:,k+1);
    S(:,:,k+1) = exp(logprice*Q').*S0';
end
