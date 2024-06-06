clear, clc, close all
% price a arithmetic basket put under Dupire's local volatility
% 
% Reference:
% [1] Yang and Li (2024). Gradient-enhanced sparse Hermite polynomial
% expansions for pricing and hedging high-dimensional American options
% [2] Becker, S., Cheridito, P., Jentzen, A., & Welti, T. (2021). 
% Solving high-dimensional optimal stopping problems using deep learning. 

addpath ../utils
%%% set parameters
p.strike = 100;
p.rate = 0.05; 
p.dividend = 0.1;
p.expiration = 1;
p.dim = 5;                                   % asset number
p.S0 = 100*ones(p.dim,1);
% p.volatility = 
p.correlation = eye(p.dim);
p.numTimeStep = 48;                     

M = 200000;
order = 15;                                  % polynomials up to the order 
I = hyperbolic_cross_indices(p.dim, order);   % generate hyperbolic cross index set
Nbasis = size(I,1);


% under construction ...
