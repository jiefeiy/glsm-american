clear, clc

% I = max b1 + b2 + ... + bd
% s.t. b1 * b2 * ... * bd <= p+1

d = 3;
order = 6;
I = hyperbolic_cross_indices(d, order)
max(sum(I, 2))