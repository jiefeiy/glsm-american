function [W, X] = gen_paths_heston_logscale(p, M)
% X stores the paths of log-price and log-variance
d = 2;
v0 = p.v0;
r = p.rate;
di = p.dividend;
rho = p.rho;
kappa = p.kappa;
theta = p.theta;
nu = p.nu;
T = p.expiration;
N = p.numTimeStep;
dt = T/N;

W = zeros(M,d,N);
X = zeros(M,d,N);

for k = 0:N-1
    if k == 0
        W(:, :, 1) = sqrt(dt)*randn(M,d);
        X(:,1,1) = (r-di-.5*v0)*dt + sqrt(v0)*( rho*W(:,1,1) + sqrt(1-rho^2)*W(:,2,1) );
        vol = abs( v0 + kappa * (theta - v0) * dt + nu *sqrt(v0) * W(:,1,1) );
        X(:,2,1) = log(vol);
    else
        dW = sqrt(dt)*randn(M,d);
        W(:, :, k+1) = W(:,:,k) + dW;
        sqrtvol = sqrt(vol);
        X(:,1,k+1) = X(:,1,k) + (r-di-.5*vol)*dt +...
            sqrtvol.*( rho*dW(:,1) + sqrt(1-rho^2)*dW(:,2) );
        vol = abs( vol + kappa*(theta - vol)*dt + nu * sqrtvol .* dW(:,1) );
        X(:,2,k+1) = log(vol);
    end
end


% % direct Euler method for log-variance 
% for k = 0:N-1
%     if k == 0
%         W(:, :, 1) = sqrt(dt)*randn(M,d);
%         X(:,1,1) = (r-di-.5*v0)*dt + sqrt(v0)*( rho*W(:,1,1) + sqrt(1-rho^2)*W(:,2,1) );
%         X(:,2,1) = log(v0) + ((kappa*theta-.5*nu^2)/v0 - kappa)*dt + nu/sqrt(v0)*W(:,2,1);
%     else
%         dW = sqrt(dt)*randn(M,d);
%         W(:, :, k+1) = W(:,:,k) + dW;
%         X(:,1,k+1) = X(:,1,k) + (r-di-.5*exp(X(:,2,k)))*dt +...
%             exp(X(:,2,k)/2).*( rho*dW(:,1) + sqrt(1-rho^2)*dW(:,2) );
%         X(:,2,k+1) = X(:,2,k) + ((kappa*theta-.5*nu^2)*exp(-X(:,2,k)) - kappa)*dt...
%             + nu*exp(-X(:,2,k)/2).*dW(:,1);
%     end
% end
