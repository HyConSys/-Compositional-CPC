# -Compositional Control-Parity-Certificates
Simulation code corresponding to  "Control Synthesis for Large-Scale Systems via Parity Certificates" by Felipe Galarza-Jimenez and Majid Zamani

The complete comparison was produced by ```Benchmark_CPC.jl```; the tuned certificate was independently
reproduced with ```Synthesis_SOS.jl 2 100 optimized 0.5 deterministic```. 

The runs used: 
* Julia 1.10.8,
* JuMP 1.30.1, 
* MOSEK 11.2.2, 
* MosekTools 0.15.10, 
* TSSOS source tree at revision bfe61d6.
