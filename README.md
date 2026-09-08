# Compositional Control Parity Certificates

Simulation code for the paper *Control Synthesis for Large-Scale Systems via Parity Certificates* by Felipe Galarza-Jimenez and Majid Zamani.

The code uses sum-of-squares (SOS) optimization to synthesize and verify control parity certificates.

## Repository contents:
- `Benchmark_CPC.jl` — runs the complete homogeneous, monolithic, and heterogeneous comparison.
- `Synthesis_SOS.jl` — reproduces the tuned homogeneous compositional certificate.
- `CPC_SOS_core.jl` — contains the shared SOS formulation and utilities.
- `Synthesis_Monolithic_SOS.jl` — contains the monolithic formulation used by the benchmark.
- `Synthesis_Heterogeneous_SOS.jl` — contains the heterogeneous formulation used by the benchmark.

## Software

The reported runs used:
- Julia 1.10.8,
- JuMP 1.30.1,
- MOSEK 11.2.2,
- MosekTools 0.15.10, and
- TSSOS source tree at revision `bfe61d6`.
- Include `DynamicPolynomials` and `MultivariatePolynomials`.

## Reproducing the results

The complete comparison is produced by:

```bash
julia --startup-file=no Benchmark_CPC.jl
```
This command writes `CPC_benchmark_results.tsv`.

The tuned degree-two certificate for the homogeneous interconnection with `N = 100`, optimized supply matrices, `alpha = 0.5`, and deterministic local transitions is independently reproduced by:

```bash
julia --startup-file=no Synthesis_SOS.jl 2 100 optimized 0.5 deterministic
```
This command writes `CPC_SOS_results_N100_degree2_alpha0p5_deterministic.txt`.

