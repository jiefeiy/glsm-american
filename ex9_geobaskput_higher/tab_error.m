clear, clc
addpath ../utils/
%%% load reference solution
file_name_ref = 'geobaskput_ref_interp_d20_30_40_50';
load(['./data/' file_name_ref])
V0_ref = V0_vals;
clear V0_vals

%%% d = 20
p.dim = 20;                                          % asset number
M = 100000;
order = 10;
I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

num_trials = 10; 
file_name = ['geobaskput_GLSM_d' num2str(p.dim) '_M' num2str(M)...
    '_order' num2str(order) '_Nb' num2str(Nbasis) '_trials' num2str(num_trials)];

load(['./data/' file_name])

V20 = mean(V0_vals)
abs(V20 - V0_ref(1)) / V0_ref(1)


%%% d = 30
p.dim = 30;                                          % asset number
M = 100000;
order = 10;
I = hyperbolic_cross_indices(p.dim, order);
Nbasis = size(I,1);

num_trials = 5; 
file_name = ['geobaskput_GLSM_d' num2str(p.dim) '_M' num2str(M)...
    '_order' num2str(order) '_Nb' num2str(Nbasis) '_trials' num2str(num_trials)];

load(['./data/' file_name])

V30 = mean(V0_vals)
abs(V30 - V0_ref(2)) / V0_ref(2)





