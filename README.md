# Compositional Control-Parity Certificates

Simulation code for the paper *Control Synthesis for Large-Scale Systems via
Parity Certificates* by Felipe Galarza-Jimenez and Majid Zamani.

The released experiments use degree-two certificate templates for fixed
controller candidates. Every reported compositional certificate is checked with exact rational
arithmetic, including the aggregate supply inequality. The dense monolithic
SOS run is a floating-point numerical computation; a separate rational
certificate establishes the monolithic guarantee.

## Validated cases

All cases use alpha = 0.01 and the physical actuator set U = [0,3], where
u = 0 means that the heater is off. The compact state interval [15,30] is a
certification domain, and a selected controller value is admissible when its
successor remains in this interval. The fixed controllers below satisfy this
condition over the internal-input domains used by the certificates.

- Homogeneous compositional rings with N = 10 and N = 100:
  eta = 0.1 and u = 2.
- Homogeneous monolithic system with N = 10:
  eta = 0.1 and u = 2.
- Heterogeneous compositional ring with 100 scalar rooms:
  twenty two-room clusters use eta = 0.1 and u = 2, and twenty three-room
  clusters use eta = 0.05 and u = 2.6.

The global good label means that every room lies in [23,26]. The global bad
label means that at least one room is outside [23,26].

## Coordinate convention

The simulations evolve the original physical temperatures. For the homogeneous
certificate, the verifier uses the formally equivalent shifted-port realization

    z = x - 25,    v = w - 50,

so the physical state certificates are `V_q0(x) = 2(x-25)^2 + 1` and
`V_q1(x) = 2(x-25)^2`. The supply matrix acts on `[w-50, x-25]`, not on
`[w, x]`. Because the equilibrium satisfies `w_bar = M*y_bar`, the centered
interconnection remains `v = M*z`. Equivalently, the state remains `x`, the
output is `h_tilde(x)=x-25`, and the internal-input form of the transition is
`f_tilde(x,u,v)=f(x,u,v+50)`. Thus this is a port reparameterization of the
original plant, not a different state dynamics or labeling convention.

The heterogeneous verifier likewise centers every cluster at its exact
period-five equilibrium, outputs the full shifted cluster state, and uses a
rectangular selector to obtain the two neighboring boundary-temperature
deviations. It evaluates the original physical transition at the shifted
internal input plus the corresponding equilibrium boundary temperatures.

The monolithic numerical SOS program uses
`xhat = (x - 22.5)/7.5`. A polynomial printed in `xhat` applies to the
physical system by this affine substitution. This pullback preserves the Gram
matrix but does not make rounded floating-point coefficients exact.

## Minimal Julia release

- CPC_Centered_ADMM.jl
- Verify_Centered_Analytic_CPC.jl
- CPC_SOS_core.jl
- Synthesis_Monolithic_SOS.jl
- Verify_Monolithic_Analytic_CPC.jl
- Verify_Heterogeneous_Intervals.jl

## Software

The reported runs used:

- Julia 1.10.8,
- JuMP 1.30.1,
- MOSEK 11.2.2,
- MosekTools 0.15.10,
- DynamicPolynomials,
- MultivariatePolynomials, and
- the TSSOS source tree at revision bfe61d6.

A valid MOSEK license is required for the generic monolithic SOS run.

## Reproducing the results

Homogeneous compositional N = 10:

    julia --startup-file=no CPC_Centered_ADMM.jl 10 2

Homogeneous compositional N = 100:

    julia --startup-file=no CPC_Centered_ADMM.jl 100 2

Monolithic N = 10 SOS:

    julia --startup-file=no Synthesis_Monolithic_SOS.jl 10 2 0.01

Independent exact monolithic verification:

    julia --startup-file=no Verify_Monolithic_Analytic_CPC.jl

Heterogeneous compositional N = 100:

    julia --startup-file=no Verify_Heterogeneous_Intervals.jl

The heterogeneous verifier also performs the matched N = 10 compositional and
monolithic check for the identical [2,3,2,3] cluster pattern.
