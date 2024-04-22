clear, clc, close all
% implement COS method for Bermudan option pricing under Heston model
% Reference:
% [1] Fang and Oosterlee (2011). A Fourier-Based Valuation Method for
% Bermudan and Barrier Options under Heston's Model.
% [2] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options.

% Copyright 2024
% Written by Jiefei Yang  - 4/2024

% using (X, sig) as variables
% X = log(S/K); sig = log(V);

% Heston SDEs:
% dX_t = (r-q-V_t/2)dt + sqrt(V_t) dB_t
% dV_t = kappa(theta - V_t)dt + nu*sqrt(V_t) dW_t

p.S0 = 9; 
p.strike = 10;
p.rate = 0.1;
p.rho    = 0.1;

% example 2 setting
p.nu     = 0.9;
p.v0     = 0.0625;
p.theta  = 0.16;
p.kappa    = 5;

p.expiration = 0.25;
p.numTimeStep = 50;

S0 = p.S0;
K = p.strike;
r = p.rate;
rho = p.rho;
nu = p.nu;
v0 = p.v0;
theta = p.theta;
kappa = p.kappa;
T = p.expiration;
N = p.numTimeStep;

dt = p.expiration./p.numTimeStep;
disc = exp(-p.rate*dt);

%% define kernel function using ChF
q = (2*kappa*theta)/(nu^2) - 1;
zeta = 2*kappa/(nu^2*(1-exp(-kappa*dt)));
gf = @(u) sqrt( kappa^2 - 2*1i*nu^2*u );
u = @(w) w*(rho*kappa/nu - .5) + .5*1i*w.^2*(1-rho^2);
% (eq. 29) in the paper by F Fang
chf_need = @(w, st, ss) zeta*exp(    1i*w*( r*dt + rho/nu*( exp(st) - exp(ss) - kappa*theta*dt ) ) ...
       + abs( real( exp((st + ss)/2).* 4*gf(u(w)).*exp( -.5*gf(u(w))*dt )./nu^2./(1-exp( -gf(u(w))*dt )) ) ) ...
       -.5*(gf(u(w)) - kappa)*dt + ...
       (exp(ss) + exp(st))/(nu^2).*(  kappa*(1+exp(-kappa*dt))./(1-exp(-kappa*dt)) - gf(u(w)).*(1+exp(-gf(u(w))*dt))./(1-exp(-gf(u(w))*dt))  )...
       -zeta.*( exp(ss)*exp(-kappa*(dt)) + exp(st) )    ).*...
    (exp(st)./(exp(ss).*exp(-kappa*(dt)))).^(q/2).*exp(st) ...
    .*besseli(q, exp((st + ss)/2).* 4*gf(u(w)).*exp( -.5*gf(u(w))*dt )./nu^2./(1-exp( -gf(u(w))*dt )), 1  ) ...
    .*gf(u(w)).*(1-exp(-kappa*dt))./kappa./(1-exp( -gf(u(w))*dt ));

%% quadrature points
mid_v = log(v0*exp(-kappa*T) + theta*(1-exp(-kappa*T)));
q = 2*kappa*theta/(nu^2)-1;
% av = mid_v - 5/(1+q);
% bv = mid_v + 2/(1+q);
av = -7;
bv = log(0.8);
J = 2^7;
[sigknots,sigweights] = lgwt(J,av,bv);   % Legendre-Gauss Quadrature

%% initialize
a = -1; b = 1; Ncos = 2^7; 
V = CallPutCoefficients('p',a, b, a, 0, 0:Ncos-1, K)'*ones(1,J);
Psi = zeros(Ncos, J, J);
for p = 1:J
    for j = 1:J
        Psi(:, j, p) = chf_need((0:Ncos-1)'*pi/(b-a), sigknots(j), sigknots(p));
    end
end
beta = zeros(Ncos, J);
for p = 1:J
    beta(:,p) = (V.*Psi(:,:,p))*sigweights;
end
beta(1,:) = .5*beta(1,:);   % halve the first row

%% main loop
for k = N-1:-1:1    
    % find the early-exercise point
    xstar = zeros(J,1);
    for p = 1:J
        fun = @(x) real( exp(1i*(0:Ncos-1)*pi*(x-a)/(b-a)) * beta(:,p) )*disc - max(K - K*exp(x), 0);
        xstar(p) = fzero(fun,0);
    end
    
    % calculate the cos coefficients of value function
    C = zeros(Ncos, J); V = C;
    for p=1:J
        Mc =   generate_hankel(a,b,xstar(p),b,Ncos);
        Ms = generate_toeplitz(a,b,xstar(p),b,Ncos);
        C(:,p) = disc/pi*imag( ( Mc+Ms )*beta(:,p) );
        V(:,p) = C(:,p) + CallPutCoefficients('p',a,b,a,xstar(p),0:Ncos-1,K)';
    end 

    beta = zeros(Ncos, J);
    for p = 1:J
        beta(:,p) = (V.*Psi(:,:,p))*sigweights;
    end
    beta(1,:) = .5*beta(1,:);   % halve the first row

    % plot the continuation value function
    if k == 25
        Vk = zeros(J,50); Xtest = linspace(a,b,50);
        for p = 1:J
            for i = 1:50
                Vk(p,i) = real( exp(1i*(0:Ncos-1)*pi*(Xtest(i)-a)/(b-a)) * beta(:,p) )*disc;
            end
            hold on;
            plot3(log(K/S0) + Xtest, (sigknots(p))*ones(size(Xtest)), max(K - K*exp(Xtest), 0), '.', 'Color','b');
            plot3(log(K/S0) + Xtest, (sigknots(p))*ones(size(Xtest)), Vk(p,:), '.', 'Color','r');
        end
%         plot3(xstar, sigknots, max(K - K*exp(xstar), 0), 'ob');
    end

    disp(k)
end

x0 = log(S0/K);
V0_data = zeros(J,1);
for p = 1:J
    V0_data(p) = real( exp(1i*(0:Ncos-1)*pi*(x0-a)/(b-a)) * beta(:,p) )*disc;
end
V0 = interp1(sigknots,V0_data,log(v0),'spline')
xlabel('log-price');
ylabel('log-variance')

%% function dependence
function Mc = generate_hankel(a,b,x1,x2,Ncos)
% define Hankel matrix
    c = [1i*pi*(x2 - x1)/(b-a)   1./(1:Ncos-1).*( exp(1i*(1:Ncos-1)*(x2-a)*pi/(b-a)) - exp(1i*(1:Ncos-1)*(x1-a)*pi/(b-a)) )];
    r = 1./((Ncos-1):(2*Ncos-2)).*( exp(1i*((Ncos-1):(2*Ncos-2))*(x2-a)*pi/(b-a)) - exp(1i*((Ncos-1):(2*Ncos-2))*(x1-a)*pi/(b-a)) );
    Mc = hankel(c,r);
end

function Ms = generate_toeplitz(a,b,x1,x2,Ncos)
% define Toeplitz matrix
    c = [1i*pi*(x2-x1)/(b-a)   1./(-(1:Ncos-1)).*( exp(1i*(-(1:Ncos-1))*(x2-a)*pi/(b-a)) - exp(1i*(-(1:Ncos-1))*(x1-a)*pi/(b-a)) )];
    r = [1i*pi*(x2-x1)/(b-a)   1./(1:Ncos-1).*( exp(1i*(1:Ncos-1)*(x2-a)*pi/(b-a)) - exp(1i*(1:Ncos-1)*(x1-a)*pi/(b-a)) )];
    Ms = toeplitz(c,r);
end

function G_k = CallPutCoefficients(CP,a,b,l,u,k,K)
% K is the strike price
    if lower(CP) == 'c' || CP == 1
        c = max(l,0);
        d = max(u,0);
        [Chi_k,Psi_k] = Chi_Psi(a,b,c,d,k);
        if a < b && b < 0.0
            G_k = zeros([length(k),1]);
        else
            G_k = 2.0 / (b - a)*K * (Chi_k - Psi_k);
        end
    elseif lower(CP) == 'p' || CP == -1
        c = min(l,0);
        d = min(u,0);
        [Chi_k,Psi_k] = Chi_Psi(a,b,c,d,k);
        G_k = 2.0 / (b - a)*K * (- Chi_k + Psi_k);
    end
end

function [chi_k,psi_k] = Chi_Psi(a,b,c,d,k)
psi_k = sin(k * pi * (d - a) / (b - a)) - sin(k * pi * (c - a)/(b - a));
psi_k(2:end) = psi_k(2:end) * (b - a) ./ (k(2:end) * pi);
psi_k(1) = d - c;
chi_k = 1.0 ./ (1.0 + (k * pi / (b - a)).^2);
expr1 = cos(k * pi * (d - a)/(b - a)) * exp(d) ...
    - cos(k * pi * (c - a) / (b - a)) * exp(c);
expr2 = k * pi / (b - a) .* sin(k * pi * (d - a) / (b - a)) *exp(d) ...
    - k * pi / (b - a) .* sin(k * pi * (c - a) / (b - a)) * exp(c);
chi_k = chi_k .* (expr1 + expr2);
end

function [x,w]=lgwt(N,a,b)

% lgwt.m
%
% This script is for computing definite integrals using Legendre-Gauss 
% Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
% [a,b] with truncation order N
%
% Suppose you have a continuous function f(x) which is defined on [a,b]
% which you can evaluate at any x in [a,b]. Simply evaluate it at all of
% the values contained in the x vector to obtain a vector f. Then compute
% the definite integral using sum(f.*w);
%
% Written by Greg von Winckel - 02/25/2004
N=N-1;
N1=N+1; N2=N+2;

xu=linspace(-1,1,N1)';

% Initial guess
y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2);

% Legendre-Gauss Vandermonde Matrix
L=zeros(N1,N2);

% Derivative of LGVM
Lp=zeros(N1,N2);

% Compute the zeros of the N+1 Legendre Polynomial
% using the recursion relation and the Newton-Raphson method

y0=2;

% Iterate until new points are uniformly within epsilon of old points
while max(abs(y-y0))>eps
    
    
    L(:,1)=1;
    Lp(:,1)=0;
    
    L(:,2)=y;
    Lp(:,2)=1;
    
    for k=2:N1
        L(:,k+1)=( (2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1) )/k;
    end
 
    Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);   
    
    y0=y;
    y=y0-L(:,N2)./Lp;
    
end

% Linear map from[-1,1] to [a,b]
x=(a*(1-y)+b*(1+y))/2;      

% Compute the weights
w=(b-a)./((1-y.^2).*Lp.^2)*(N2/N1)^2;
end