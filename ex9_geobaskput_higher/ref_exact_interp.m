clear, clc, close all
% => For multi-dimensional American geometric basket put benchmark. 
% discounted to present first
% CC quadrature + barycentric formula at Chebyshev-Lobatto pts

d_vals = [20 30 40 50];
V0_vals = zeros(size(d_vals));
i = 1;
for d = d_vals
    V0_vals(i) = run_1d_geo_bask_put(d);
    fprintf("The reference Bermudan price of d = %d is %1.5f \n", d, V0_vals(i));
    fprintf("--------------------------------------------------\n");
    i = i+1;
end

file_name = 'geobaskput_ref_interp_d20_30_40_50';
save(['data/' file_name '.mat']);


function V0 = run_1d_geo_bask_put(d)
S0 = 100; r = 0.03; sig = 0.2./d*sqrt(0.5*d^2 + 0.5*d); 
DivYield = 0.2^2/2 - sig^2/2;
T = 0.25; kappa = 100; K = 50;

M = 2000;                                 % number of interpolation pts
n = 2000;                                 % number of quadrature pts
dt = T/K;
NumInterp = 5;                           % NumInterp standard error of interpolation interval
NumSig = 6;                              % NumSig standard error of quadrature interval
lb = log(S0) + (r-DivYield-sig*sig/2)*T - NumInterp*sig*sqrt(T);
ub = log(S0) + (r-DivYield-sig*sig/2)*T + NumInterp*sig*sqrt(T);

x = (ub-lb)/2*cos( pi*(0:M)/M ) + (ub+lb)/2; % interpolation pts x
C = zeros(size(x)); s = zeros(n+1, M+1);

% compute quadrature pts s(:)
for m = 1:M+1
    li = x(m) + (r-DivYield-sig*sig/2)*dt - NumSig*sig*sqrt(dt);
    ui = x(m) + (r-DivYield-sig*sig/2)*dt + NumSig*sig*sqrt(dt);
    t = pi*(0:n)'/n;
    s(:,m) = (ui-li)/2*cos(t) + (ui+li)/2;
end
V = max(kappa - exp(s(:)), 0);
V = exp(-r*T)*reshape(V, n+1, M+1);

% Backward induction
for k = K-1:-1:1
    for m = 1:M+1
        li = x(m) + (r-DivYield-sig*sig/2)*dt - NumSig*sig*sqrt(dt);
        ui = x(m) + (r-DivYield-sig*sig/2)*dt + NumSig*sig*sqrt(dt);
        t = pi*(0:n)'/n;
        ftemp = 1/sig/sqrt(2*pi*dt)*exp( -(s(:,m) - x(m) - (r-DivYield-sig*sig/2)*dt).^2/(2*dt*sig*sig) );
        C(m) = clenshaw_curtis(s(:,m), V(:,m).*ftemp, li, ui);
    end
    f = barycentric_1(x', C', lb, ub, s(:));
    g = exp(-r*k*dt)*(kappa - exp(s(:)));
    V = max([g, f], [], 2);
    V = reshape(V, n+1, M+1);
end

% compute the initial value
for m = 1:M+1
    li = x(m) + (r-DivYield-sig*sig/2)*dt - NumSig*sig*sqrt(dt);
    ui = x(m) + (r-DivYield-sig*sig/2)*dt + NumSig*sig*sqrt(dt);
    t = pi*(0:n)'/n;
    ftemp = 1/sig/sqrt(2*pi*dt)*exp( -(s(:,m) - x(m) - (r-DivYield-sig*sig/2)*dt).^2/(2*dt*sig*sig) );
    C(m) = clenshaw_curtis(s(:,m), V(:,m).*ftemp, li, ui);
end
V0 = barycentric_1(x', C', lb, ub, log(S0));
end


function I = clenshaw_curtis(x,fx,xmin,xmax) % (n+1)-pt C-C quadrature
n = size(x,1)-1;
scale = (xmax - xmin)/2;
fx = fx/(2*n);                            
g = real(fft(fx([1:n+1 n:-1:2])));        % Fast Fourier Transform
a = [g(1); g(2:n)+g(2*n:-1:n+2); g(n+1)]; % Chebyshev coefficients
w = 0*a'; w(1:2:end) = 2./(1-(0:2:n).^2); % weight vector
I = scale*w*a;                            % the integral
end


function f = barycentric_1(xs, ys, lb, ub, x)
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
f(find(x<lb | x>ub)) = 0;                   % set extrapolant to zero
end



