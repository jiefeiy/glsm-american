# glsm-american

This repository includes numerical examples for the paper [Gradient-enhanced sparse Hermite polynomial expansions for pricing and hedging high-dimensional American options](https://arxiv.org/abs/2405.02570) by Jiefei Yang and Guanglian Li, 2024. 

## Quick start
To see how G-LSM method works in a 2-d max-call example, git clone this repo, and run `./quick_start_glsm.m`. 

## Reproducibility of the numerical examples
Parameters used in examples are listed as follows. 
|Example  | Parameters|
|-------- | ----------|
|1. Geometric basket put  | $K = 100, T = 0.25, r = 0.03, \delta_i = 0, \sigma_i = 0.2,\rho_{ij} = 0.5, N = 50$|
|2. Geometric basket call | $K = 100, T = 2, r = 0, \delta_i = 0.02, \sigma_i = 0.25, \rho_{ij} = 0.75, N=50$|
|3. Max-call with $d$ symmetric assets | $K = 100, T = 3, r = 0.05, \delta_i = 0.1, \sigma_i = 0.2, \rho_{ij} = 0, N=9$|
|4. Max-call with $d$ asymmetric assets |  If $d\le 5$, $\sigma_i = 0.08 + 0.32\times(i-1)/(d-1)$; if $d>5$, $0.1 + i/(2d)$|
|5. Put option under Heston model | $K=10, T=0.25, r=0.1, v_0 = 0.0625, \rho = 0.1, \kappa = 5, \theta = 0.16, \nu = 0.9, N=50$|

- `./ex1_geobaskput/` includes tests for example 1.
- `./ex2_geobaskcall/` includes tests for example 2.
- `./ex3_maxcall_sym/` includes tests for example 3.
- `./ex4_maxcall_asym/` includes tests for example 4.
- `./ex5_heston/` includes tests for example 5.

## Other methods
- Least squares Monte Carlo (LSM): `./ex1_geobaskput/lsm_geobaskput.m` tests with LSM. Longstaff, F. and Schwartz, E. (2001). Valuing American
options by simulation: a simple least-squares approach.
- Cosine method (COS) under Heston model: `./ex5_heston/cos_heston.m` tests with COS. Fang, F. and Oosterlee, C. W. (2011). A Fourier-based valuation
method for Bermudan and barrier options under Heston’s model.

## Citation
```
@article{yang2024gradient,
  title={Gradient-enhanced sparse Hermite polynomial expansions for pricing and hedging high-dimensional American options},
  author={Yang, Jiefei and Li, Guanglian},
  journal={arXiv preprint arXiv:2405.02570},
  year={2024}
}
```

