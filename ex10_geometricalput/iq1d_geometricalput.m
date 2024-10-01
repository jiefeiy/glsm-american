%%% set parameters
p.strike = 1; p.rate = 0.05; p.dividend = 0;
p.expiration = 1;
p.dim = 40;                                          % asset number
p.S0 = ones(p.dim,1);
p.volatility = eye(p.dim)*0.2;
p.correlation = eye(p.dim);
p.numTimeStep = 10;
p.callput = 'put';

% discretization parameters
numInterp = 5;         % NumInterp standard error of truncated interpolation interval
nq = 512;                % number of quadrature pts
M = 1024;               % number of interpolation pts

% reduce to 1d problem
d = p.dim;
K = p.strike; 
r = p.rate * d; 
T = p.expiration;
% S0 = geomean(p.S0);
S0 = prod(p.S0);
% sig = sqrt( diag(p.volatility)' * p.correlation * diag(p.volatility) )/d;
sig = 0.2 * sqrt(d);
% di = mean(p.dividend + diag(p.volatility).^2/2) - sig^2/2;
di = 0;
N = p.numTimeStep;
dt = T/N;
disc = exp(-p.rate*dt);

% quadrature knots and weights
[qknots,qweights] = knots_normal(nq,0,1);

% determine the truncated interpolation interval
mu = r-di-sig^2/2;
lb = mu*T - numInterp*sig*sqrt(T);
ub = mu*T + numInterp*sig*sqrt(T);

x = (ub-lb)/2*cos( pi*(0:M-1)'/(M-1) ) + (ub+lb)/2; % interpolation pts x
xq = zeros(M, nq);
for m = 1:M
    xq(m,:) = x(m) + mu*dt + sqrt(dt)*sig*qknots';
end
xstar = zeros(N-1,1);

%%% dynamic programming
switch p.callput
    case 'put'
        payoff = max(K - S0*exp(xq), 0);
    case 'call'
        payoff = max(S0*exp(xq) - K, 0);
end
EV = payoff;
for k = N-1:-1:1

    CV_pts = payoff*qweights'*disc;
    CV = barycentric(x, CV_pts, lb, ub, xq(:));
    CV = reshape(CV, size(xq));
    payoff = max(EV, CV);
    
    switch p.callput
        case 'put'
            fun = @(xx) barycentric(x, CV_pts, lb, ub, xx) - (K - S0*exp(xx));
        case 'call'
            fun = @(xx) barycentric(x, CV_pts, lb, ub, xx) - (S0*exp(xx) - K);
    end
    xstar(k) = fzero(fun, 0);            % find the critical value
    
%     idx = EV(:) > CV(:);
%     t = k*dt;    
%     scatter(t*ones(size(find(idx))) , S0*exp(xq(idx)), 3, 'b', 'filled'); hold on;
%     scatter(t*ones(size(find(~idx))) , S0*exp(xq(~idx)), 3, 'r', 'filled'); hold on;
%     scatter(t, S0*exp(xstar(k)), '*');

end
CV_pts = payoff*qweights'*disc;

Sstar = S0*exp(xstar);
V0 = barycentric(x, CV_pts, lb, ub, 0)


%----------------------------------------------------------------------
function [x,w]=knots_normal(n,mi,sigma)
if (n==1) 
      % the point (traslated if needed) 
      x=mi;
      % the weight is 1:
      w=1;
      return
end

% calculates the values of the recursive relation
[a,b]=coefherm(n); 

% builds the matrix
JacM=diag(a)+diag(sqrt(b(2:n)),1)+diag(sqrt(b(2:n)),-1);

% calculates points and weights from eigenvalues / eigenvectors of JacM
[W,X]=eig(JacM); 
x=diag(X)'; 
w=W(1,:).^2;
[x,ind]=sort(x); %#ok<TRSRT>
w=w(ind);

% modifies points according to mi, sigma (the weigths are unaffected)
x=mi + sqrt(2)*sigma*x;
end


%----------------------------------------------------------------------
function [a, b] = coefherm(n)
if (n <= 1)
    disp(' n must be > 1 '); 
    return; 
end
a=zeros(n,1); 
b=zeros(n,1); 

b(1)=sqrt(pi);
k=2:n;
b(k)=0.5*(k-1); 
end

function f = barycentric(xs, ys, lb, ub, x)
% Chebyshev interpolation using barycentric interpolation formula
% Chebyshev pts (2nd kind) / Chebyshev-Lobatto points
% -------------------------------------------------------------
% xs     =   Chebyshev pts                            (m-by-1 vector)
% ys     =   function value at interpolation knots    (m-by-1 vector)
% lb     =   lower bound of the interpolant interval          (float)
% ub     =   upper bound of the interpolant interval          (float)
% x      =   the points needed to evaluate                   (vector)
% --------------------------------------------------------------
% f      =   interpolation value at the points x             (vector)
% ---------------------------------------------------------------
m = size(xs,1) - 1; % number of interpolation points-1
% Barycentric interpolation
c =  [1/2; ones(m-1,1); 1/2].*(-1).^((0:m)');
numer = zeros(size(x));
denom = zeros(size(x));
exact = zeros(size(x));
for j = 1:m+1
    xdiff = x-xs(j);
    temp = c(j)./xdiff;
    numer = numer + temp*ys(j);
    denom = denom + temp;
    exact(xdiff==0) = j;
end
f = numer./denom; 
jj = find(exact); f(jj) = ys(exact(jj));
f(x<lb | x>ub) = 0;                   % set extrapolant to zero
end
