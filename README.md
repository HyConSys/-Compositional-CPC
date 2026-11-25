# Compositional control-parity certificates for buck-converter networks

Simulation code for **Control Synthesis for Large-Scale Systems via Parity
Certificates**, by Felipe Galarza-Jimenez and Majid Zamani.

This project compares monolithic and compositional **controller-and-certificate
synthesis** on heterogeneous, resistively coupled buck-converter networks.
The comparison routine uses $N\in\{2,10,50,100\}$, with the same plant,
controller family, polynomial degree and solver settings for both methods.

The specification is target availability: eventually, at every subsequent
sample, **at least one converter is inside its target disk**. The converter
providing availability may change. The property does not require every
converter to reach its target.

## Implementation

The release contains **one Python routine and three Julia implementation files**.
All dynamics, controller LPs, monolithic/compositional SOS problems, supply
coordination and exact certificate verification are implemented in Julia.
Python schedules the network sizes and methods, supervises time and memory,
and collects the comparison results.

| File | Implementation |
| --- | --- |
| [Compare_Direct_CPC.py](Compare_Direct_CPC.py) | Runs the comparison for $N=2,10,50,100$, supervises isolated Julia processes, and produces CSV/Markdown summaries. |
| [Synthesis_Direct_CPC.jl](Synthesis_Direct_CPC.jl) | Implements controller candidate generation, the monolithic and compositional CPC SDPs, rational repair, and physical-coordinate certificate exports. |
| [Synthesis_Buck_CPC.jl](Synthesis_Buck_CPC.jl) | Implements the converter network, signal selectors, solver configuration, clique construction, and exact rational PSD checks. |
| [Benchmark_Direct_CPC.jl](Benchmark_Direct_CPC.jl) | Executes each Julia synthesis problem, records computational stages, verifies saved witnesses in a fresh process, and displays B,V as readable polynomials. |

In `Synthesis_Direct_CPC.jl`, `add_monolithic_conditions!` constructs the global
SOS conditions; `add_compositional_conditions!` and `add_local_conditions!`
construct the local problems and aggregate supply constraints. The algorithms
are described mathematically below.

## Software and solvers

The reported computations used:

| Component | Version or role |
| --- | --- |
| Julia | 1.10.8 |
| JuMP | 1.30.1; LP and SDP modeling |
| MOSEK | 11.2.2; solver for both formulations |
| MosekTools | 0.15.10; JuMP interface to MOSEK |
| Clarabel | 0.11.1; available in the shared solver utility; not used in the reported comparison |
| Python | 3.11 or newer; standard library only |

Julia uses exact `Rational{BigInt}` arithmetic and its standard linear-algebra,
random-number, TOML, date and SHA utilities. A valid MOSEK license is required.
The Python resource supervisor requires Linux for process-group memory checks.
The runtime dependencies are declared in [Project.toml](Project.toml) and pinned
in [Manifest.toml](Manifest.toml).

## Running the comparison

Initialize the Julia project:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Prepare the four network sizes without starting a simulation:

```bash
python3 Compare_Direct_CPC.py prepare --sizes 2 10 50 100 \
  --seed 20261122 --candidates 3 --progress-policy auto \
  --memory-gib 8 --search-timeout 600 --worker-timeout 900 --solver-timeout 120
```

Preparation reports the generated campaign identifier and freezes the source
and dependency files. Replace `CAMPAIGN` below with that identifier:

```bash
python3 Compare_Direct_CPC.py run CAMPAIGN
```

Alternatively, `launch` starts a background supervisor and a separate watcher:

```bash
python3 Compare_Direct_CPC.py launch CAMPAIGN
python3 Compare_Direct_CPC.py status CAMPAIGN
python3 Compare_Direct_CPC.py summarize CAMPAIGN
```

Use either `run` or `launch` once per prepared campaign. The launcher invokes
Julia with the frozen project, so each worker uses the declared dependencies.
For another measurement, prepare a new campaign.

An unexpected model, source or proof inconsistency stops the routine. If one
method reaches a time or memory limit, the other is still attempted. If both
reach their resource limits at the same network size, the routine stops there.
Candidate-budget exhaustion is recorded as `NO_CERTIFICATE`, without claiming
nonexistence. Reports distinguish controller search, rejected attempts, SOS
construction, SOS solution, exact verification, and independent reload.

### Run one network size with one method

Run the Julia worker directly to select **one** method and **one** network
size. Execute these commands from the project directory, after initializing
the Julia project as above.

For compositional synthesis at `N=10`:

```bash
julia --project=. Benchmark_Direct_CPC.jl work \
  10 compositional Results/N10_compositional 20261122 3 120 auto
```

After a successful run, verify its exported certificate in a fresh process:

```bash
julia --project=. Benchmark_Direct_CPC.jl verify \
  Results/N10_compositional/result.toml Results/N10_compositional/Reload
```

For monolithic synthesis at the same size:

```bash
julia --project=. Benchmark_Direct_CPC.jl work \
  10 monolithic Results/N10_monolithic 20261122 3 120 auto
```

Then verify its exported certificate:

```bash
julia --project=. Benchmark_Direct_CPC.jl verify \
  Results/N10_monolithic/result.toml Results/N10_monolithic/Reload
```

The positional arguments after `work` are:

| Argument | Example | Meaning |
| --- | --- | --- |
| Network size | `10` | Any integer `N >= 2`; replace this and the output-directory name to test another size. |
| Method | `compositional` | Either `compositional` or `monolithic`. |
| Output directory | `Results/N10_compositional` | Destination for this run; choose a fresh directory for each experiment. |
| Seed | `20261122` | Physical-parameter seed used for the reported comparison. |
| Candidate budget | `3` | Maximum number of controller candidates. |
| SOS solver limit | `120` | Seconds allowed per SOS optimization call. |
| Progress policy | `auto` | Matches the published policy: uniform local steps when all equilibria are inside their targets, otherwise a nonnegative allocation with positive total. |

Keep `auto` explicit: the worker's omitted-policy default is `uniform`, which
can exclude valid availability certificates, including the reported `N=10`
compositional solution. `allocated` is another supported policy; it assigns
the total step to one eligible converter even if all equilibria are inside.

The worker first warms up the **selected method** on a separate `N=2` plant;
the requested experiment then uses the chosen `N` and seed. Its output
contains `progress.toml`, `outcome.toml` and, on success, `result.toml` with
the exact controller and certificate. The `verify` command checks the complete
saved witness and physical vertex bounds. If synthesis does not produce a
successful witness, inspect `outcome.toml` before attempting verification.

Direct workers apply the per-SOS limit above and the 120-second per-controller-LP
limit. The Python supervisor provides the separate whole-search and memory
guards described earlier; those guards do not apply to a direct Julia command.
The top-level command for `Synthesis_Direct_CPC.jl` runs both methods, so use
`Benchmark_Direct_CPC.jl work` for a single method with the published progress
policy. To compare both methods at just one size under the Python supervisor,
replace `--sizes 2 10 50 100` by `--sizes 10` in the preparation command.

## 1. Plant, domains and interconnection

For any $N\ge2$, define the undirected cycle

$$
\mathcal E_N=\{\{n,1+(n\bmod N)\}:n=1,\ldots,N\}.
$$

Edges are unordered and counted once. At $N=2$ there is one physical line.
Converter $n$ has state $x_n=(i_n,v_n)$ and duty ratio $u_n\in U_n=[0,1]$:

$$
\begin{aligned}
i_n^+ &=i_n+\frac{\tau}{L_n}(E_nu_n-r_ni_n-v_n),\\
v_n^+ &=v_n+\frac{\tau}{C_n}
\left(i_n-G_nv_n+\sum_{j\ne n}g_{nj}(v_j-v_n)\right),
\quad\tau=50\,\mu\mathrm{s}.
\end{aligned}
$$

This Euler-discretized averaged model is the defined plant. The certificate
does not extend to switching ripple or continuous intersample motion.

| Parameter | Range | Grid increment |
| --- | --- | --- |
| Inductance $L_n$ | 0.8–1.2 mH | 0.01 mH |
| Capacitance $C_n$ | 80–120 microfarads | 1 microfarad |
| Series resistance $r_n$ | 0.08–0.16 ohms | 0.01 ohms |
| Load conductance $G_n$ | 0.16–0.24 S | 0.01 S |
| Source voltage $E_n$ | 46–54 V | 1 V |
| Line conductance $g_{nj}=g_{jn}$ | 0.005–0.015 S | 0.001 S |

The reported seed is `20261122`. Both formulations use the same generated
plant at every size, checked by a physical-task fingerprint. Historical unused
gain draws are retained in the generator because deleting them changes later
random draws, including the line conductances. Those gains are ignored by
the active controller search.

Let $s_i=0.1$ A, $s_v=1$ V, $v^{\rm ref}=24$ V,
$i_n^{\rm ref}=G_nv^{\rm ref}$, and

$$
z_n=\begin{bmatrix}(i_n-i_n^{\rm ref})/s_i\\
(v_n-v^{\rm ref})/s_v\end{bmatrix}.
$$

The physical domains are

$$
\begin{aligned}
X_n&=[i_n^{\rm ref}-0.1,i_n^{\rm ref}+0.1]\times[23,25],\\
X_{0,n}&=[i_n^{\rm ref}-0.09,i_n^{\rm ref}-0.07]\times[24.6,24.8],\\
X&=\prod_nX_n,\quad X_0=\prod_nX_{0,n},\quad U=\prod_n[0,1].
\end{aligned}
$$

Thus $z_n\in[-1,1]^2$ and initially
$z_n\in[-0.9,-0.7]\times[0.6,0.8]$.

Write $d_n=\sum_{j\ne n}g_{nj}>0$ and introduce

$$
w_n^0=\sum_{j\ne n}\frac{g_{nj}}{d_n}z_{j,2}\in[-1,1],\qquad h_n^0(x_n)=z_n.
$$

The local voltage equation replaces the line-current sum by
$d_n[s_vw_n^0-(v_n-v^{\rm ref})]$. Stacking $z$ yields $w^0=Mz$, with
$M_{n,2j}=g_{nj}/d_n$ and $M_{n,2j-1}=0$.
Consequently $M[-1,1]^{2N}\subseteq[-1,1]^N$.

## 2. Shared parity automaton

Converter $n$ emits $a_n$ if $\|z_n\|_2\le r_a=0.35$, and $b_n$ otherwise.
The separable alphabet is $\prod_n\{a_n,b_n\}$.

A single DPA has initial state $q_{\rm b}$ and priorities
$\operatorname{pri}(q_{\rm b})=1$ and
$\operatorname{pri}(q_{\rm g})=2$:

| Current state | At least one $a_n$ | All $b_n$ |
| --- | --- | --- |
| $q_{\rm b}$ | $q_{\rm g}$ | $q_{\rm b}$ |
| $q_{\rm g}$ | $q_{\rm g}$ | $q_{\rm b}$ |

Acceptance requires only finitely many all-outside events. Every local problem
uses the same current global state $q$. Its exact projected successor sets are

$$
\delta_n(q,a_n)=\{q_{\rm g}\},\qquad
\delta_n(q,b_n)=\{q_{\rm b},q_{\rm g}\}.
$$

All projected successors are enforced; no independent local automaton run is
chosen. A local bad letter does not determine the global bad label because
another converter may be inside its target.

### Where the specification is encoded

The implementation specializes the paper's general framework to the two-state
DPA above. It has no runtime DPA/LTL parser or `--specification` option.
The automaton is implemented by the following functions, rather than by a
separate specification file:

| Location | Specification-dependent implementation |
| --- | --- |
| [Synthesis_Buck_CPC.jl](Synthesis_Buck_CPC.jl), `generate_instance` | Sets `radius = rat(7, 20)`, the exact target radius `0.35` in normalized coordinates. |
| [Synthesis_Direct_CPC.jl](Synthesis_Direct_CPC.jl), `target_polynomial` | Builds the target-disk domain polynomial, consistently translated to the certificate coordinates. |
| `add_monolithic_conditions!` | Creates the two automaton states, imposes the initial barrier at state 1, constructs the global label cells and successors, and applies strict rank decrease at the odd state. |
| `add_compositional_conditions!` and `add_local_conditions!` | Create the two local certificate slices and enforce every projected successor using the same global automaton state. |
| `select_progress_allocation` | Uses the availability objective to choose local progress steps from candidate-equilibrium positions. This is a search heuristic. |

In the constraint builders, array index `1` is `bad` with priority 1, and
index `2` is `good` with priority 2. Thus `-B[1]` imposes the initial-state
condition, and `q == 1` selects the strict-decrease condition. The transition
depends on the label of the **current** system state.

For centered coordinates $\xi_n=z_n-e_n$, `target_polynomial` represents

$$
g_n(\xi_n)=r_a^2-\|\xi_n+e_n\|_2^2
           =r_a^2-\|z_n\|_2^2.
$$

Inside `add_monolithic_conditions!`, the `good` array contains these disk
polynomials. The `cells` construction encodes the global transition:

```julia
cells = [(; name = "good_branch_$(n)", next = 2,
           gens = vcat(boxes, [good[n]])) for n in 1:N]
push!(cells, (; name = "bad", next = 1,
              gens = vcat(boxes, [-g for g in good])))
```

Each `good_branch` enforces the even successor on one target disk, covering
the global good union. The bad cell requires all disk polynomials to be
nonpositive. Its closed boundaries conservatively strengthen the strict
all-outside label. Both current automaton states use these cells.

Inside `add_local_conditions!`, this loop implements the exact local projections:

```julia
for q in 1:2, good in (true, false), qn in (good ? (2,) : (1, 2))
```

Here `good=true` means local letter $a_n$ and permits only successor 2;
`good=false` means local letter $b_n$ and requires inequalities for both
successors 1 and 2. The same current `q` is used by every converter. The lines
conditioned on `q == 1` subtract the global step in the monolithic rank
residual and the corresponding local step in each local rank residual.

### Changing the target or automaton

For a different radius with the **same disk-availability objective**, change
`radius` in `generate_instance`, using an exact rational such as `rat(2, 5)`
for `0.4`. All constraint builders and the progress heuristic read that value.
Run a new experiment and regenerate its witnesses and reports. A different
target shape or center requires updating `target_polynomial` and the
target-membership test in `select_progress_allocation` consistently; retain
the pullback from $z$ to $\xi$. For example, a disk centered at normalized
$\bar z_n$ uses $r_a^2-\|\xi_n+e_n-\bar z_n\|_2^2$, and the equilibrium
membership test uses $\|e_n-\bar z_n\|_2^2<r_a^2$. The present matrix
representation supports quadratic guards; higher-degree guards require
extending the polynomial representation and SOS templates.

For a **different automaton or global Boolean objective**:

1. Write down the global alphabet, label regions, initial automaton state,
   deterministic transition table and priorities. Use separable local labels
   and the shared global state required by the compositional theorem.
2. Replace the global `cells` and their `next` indices in
   `add_monolithic_conditions!`. Split unions into branches as needed and
   cover every actual label region, handling boundaries consistently.
3. Recompute each local successor set from the global transition table:
   $\delta_n(q,\sigma_n)=\{\delta(q,\sigma):\operatorname{Proj}_n(\sigma)=\sigma_n\}$.
   Update the loop in `add_local_conditions!` and enforce **all** successors
   in each set. They are determined by projection, not chosen independently.
4. Update certificate arrays, initial-state conditions and priority-dependent
   rank residuals. Additional odd priorities require the corresponding family
   of rank functions and the priority comparisons from the paper. Update the
   stationary-face reductions, which currently use the two-priority logic.
5. Revisit `select_progress_allocation` for the new objective and update
   `validate_progress_steps` if a larger rank family is introduced. A heuristic
   must not replace the required CPC inequalities or aggregate progress condition.
6. Update the descriptions in `Benchmark_Direct_CPC.jl`
   (`save_accepted_result!`), `Synthesis_Direct_CPC.jl` (`main`), and
   `Compare_Direct_CPC.py` (`prepare` and `render_comparison`), then prepare a
   fresh campaign. Recheck the SOS identities, local projections, original
   aggregate matrices and physical-coordinate bounds for the new witnesses.

For example, changing the objective to **all converters inside** changes the
monolithic good region to the intersection of all disks and the bad region
to a union of exterior branches. Its local successor sets become

$$
\delta_n(q,a_n)=\{q_{\rm b},q_{\rm g}\},\qquad
\delta_n(q,b_n)=\{q_{\rm b}\}\qquad(N\ge2).
$$

Consequently this change requires both global and local constraint changes.
The projected conditions can be more conservative, and the current local
templates/search policy may fail to find a certificate for the new objective.

The `specification` strings written to configuration/result files are
**descriptive metadata**. Editing those strings alone does not change the
mathematical problem. Existing archived witnesses belong to the original
disk-availability specification; keep them separate from new experiments.

## 3. Controller template and LP candidates

Both methods search

$$
u_n=a_n^u+
\frac{-k_{i,n}(i_n-i_n^{\rm ref})+k_{v,n}(v_n-v^{\rm ref})}{E_n},
$$

with $3N$ unknown offsets and gains. The chosen family is independent of $q$.
The reference state defines coordinates, not a prescribed equilibrium.

For numerical conditioning, the default LP variables are

$$
\begin{aligned}
t_n&=1-\tau(r_n+k_{i,n})/L_n,\\
s_n^u&=\tau(k_{v,n}-1)s_v/(L_ns_i),\\
c_n^u&=\tau(E_na_n^u-r_ni_n^{\rm ref}-v^{\rm ref})/(L_ns_i).
\end{aligned}
$$

This invertible affine change gives the exact candidate closed loop

$$
z_n^+=F_n(\theta_n)[w_n^0;z_n]+c_n(\theta_n),\qquad
F_n=\begin{bmatrix}
0&t_n&s_n^u\\
\tau d_n/C_n&\tau s_i/(C_ns_v)&1-\tau(G_n+d_n)/C_n
\end{bmatrix},\quad c_n=\begin{bmatrix}c_n^u\\0\end{bmatrix}.
$$

The admissible-controller polytope imposes

$$
\begin{aligned}
\delta_n^u&=(s_i|k_{i,n}|+s_v|k_{v,n}|)/E_n,\\
a_n^u&\ge\delta_n^u,\qquad a_n^u+\delta_n^u\le1,\\
|(c_n)_i|+\sum_{j=1}^3|(F_n)_{ij}|&\le1,\quad i=1,2.
\end{aligned}
$$

Absolute-value epigraphs make these linear constraints. They guarantee input
admissibility and robust box preservation over the entire local state and
internal-input boxes. They impose no Lyapunov decay condition and do not by
themselves imply a CPC.

One feasibility LP and $6N$ signed-coordinate support LPs explore the physical
controller coordinates. Their solutions are averaged for the first candidate;
later attempts increase one solution's weight. All support LPs are rebuilt
per attempt, as in the original timing protocol. The normalized and legacy
physical-coordinate LP modes are retained. Only the subsequent CPC feasibility
solve and exact checks accept a candidate. The default campaign tries at most
three candidates; this is an incomplete search, not joint convex co-synthesis
or CEGIS.

## 4. Translation and physical-coordinate certificates

For each candidate, assemble $z^+=A_\theta z+c_\theta$ and compute
$e=(I-A_\theta)^{-1}c_\theta$. The implementation requires an invertible
$I-A_\theta$ and $|e_i|<1$ for conditioning. It sets

$$
\xi=z-e,\qquad\omega=w^0-Me=M\xi,\qquad h_n(x_n)=\xi_n.
$$

The translated local domains are

$$
Y_n=[-1,1]^2-e_n,\qquad
W_n=[-1-(Me)_n,1-(Me)_n].
$$

The local dynamics use $w_n^0=\omega_n+(Me)_n$; target generators become
$r_a^2-\|\xi_n+e_n\|^2$. Initial boxes are translated by $-e_n$ as well.
The exact fixed-point identity cancels the drift, giving $\xi^+=A_\theta\xi$.
An independent evaluation of the original physical equations on an affine
basis checks this realization. No asymptotic-stability condition is imposed.

For $s_n=[\omega_n;\xi_{n,1};\xi_{n,2}]$, let $T_n$ select these signals
from the global $\xi$, so $s_n=T_n\xi$.
Every quadratic certificate is also pulled back to the original normalized
and physical states. The exported matrices are available under
`centered_coefficient_matrix_exact`, `original_normalized_coefficient_matrix_exact`
and `original_physical_coefficient_matrix_exact`.

## 5. CPC conditions

### Original semantic form

For a general DPA with odd priorities $\mathcal O$, a CPC consists of an
invariant function $B(x,q)$ and ranks $V_j(x,q)$, $j\in\mathcal O$.
Initially $B\le0$, and $B\le0$ implies $V_j\ge0$. On $B\le0$, one admissible
input must preserve the state domain and invariant, while

$$
V_j(x^+,q^+)\le V_j(x,q)\quad(j<\operatorname{pri}(q)),\qquad
V_k(x^+,q^+)\le V_k(x,q)-1\quad(k=\operatorname{pri}(q)\text{ odd}).
$$

The invariant is on the system--automaton product; $B$ is not generally a
defining function of the target set.

### Monolithic sufficient conditions implemented here

There is one odd priority. For $f_\theta(x)=f(x,\mu_\theta(x))$,
$q^+=\delta(q,L(x))$, common $\alpha=0.01$ and $\varepsilon=0.001N$, impose

$$
\begin{aligned}
-B_{q_{\rm b}}&\ge0 &&\text{on }X_0,\\
V_q+\alpha B_q&\ge0 &&\text{on }X,\ q\in Q,\\
\alpha B_q-B_{q^+}\circ f_\theta&\ge0 &&\text{on each label cell},\\
V_q+\alpha B_q-V_{q^+}\circ f_\theta
-\varepsilon\mathbf1_{\{q=q_{\rm b}\}}&\ge0 &&\text{on each label cell}.
\end{aligned}
$$

These are stronger sufficient conditions for the original implications.
Input and state-domain admissibility are checked separately. Scaling $B,V$
by $1/\varepsilon$ recovers unit odd-priority decrease.

### Local sufficient conditions

For each converter, use free $B_{n,q},V_{n,q}$, symmetric unrestricted
$3\times3$ matrices $\Gamma_n,\Upsilon_n$, and $\varepsilon_n\ge0$.
Initially $-B_{n,q_{\rm b}}\ge0$ and everywhere
$V_{n,q}+\alpha B_{n,q}\ge0$. For every local state, every
$\omega_n\in W_n$ and every $q^+\in\delta_n(q,L_n(x_n))$, impose

$$
\begin{aligned}
R_{B,n,q,q^+}&=\alpha B_{n,q}+s_n^\top\Upsilon_ns_n-B_{n,q^+}(x_n^+)\ge0,\\
R_{V,n,q,q^+}&=V_{n,q}+\alpha B_{n,q}+s_n^\top\Gamma_ns_n
-V_{n,q^+}(x_n^+)-\varepsilon_n\mathbf1_{\{q=q_{\rm b}\}}\ge0.
\end{aligned}
$$

One controller satisfies all applicable obligations. It cannot depend on
$\omega_n$ or the selected projected successor. The supply matrices are common
across automaton states for each converter; they may differ across converters.
Neither supply matrix is required individually to be negative semidefinite.

Composition enforces

$$
\overline\Gamma=\sum_nT_n^\top\Gamma_nT_n\preceq0,\qquad
\overline\Upsilon=\sum_nT_n^\top\Upsilon_nT_n\preceq0,\qquad
\varepsilon=\sum_n\varepsilon_n>0.
$$

Since all local inequalities use the same current priority, summation gives
strict global progress $\varepsilon$ at $q_{\rm b}$. Thus

$$
\widehat B=\frac{\sum_nB_n}{\varepsilon},\qquad
\widehat V=\frac{\sum_nV_n}{\varepsilon}
$$

satisfy the monolithic conditions. A zero local step removes only its strict
decrement, not its rank, lower bound or other transition conditions.

Under `auto`, uniform $\varepsilon_n=0.001$ is used if every local candidate
equilibrium is strictly inside its target. Otherwise the total $0.001N$ is
assigned to the closest inside-target equilibrium, with the remaining steps
zero. Candidates with no strictly inside-target equilibrium are rejected by
this heuristic. The published $N=10$ allocation is $\varepsilon_3=0.01$ and
all other steps zero. `uniform` and `allocated` remain selectable policies.

## 6. Inner optimization problems

Every $B_q,V_q$ is a free quadratic, represented by a symmetric coefficient
matrix on $[1;\xi]$. Local versions use two state variables; monolithic
versions use all $2N$. No positive-definiteness constraint is placed on these
coefficient matrices, and neither $B=0$ nor a fixed relation between ranks is
imposed.

For a domain $D=\{\zeta:g_\ell(\zeta)\ge0\}$, each residual is certified by

$$
R(\zeta)=m(\zeta)^\top Q_Rm(\zeta)+\sum_\ell\lambda_{R,\ell}g_\ell(\zeta),
\qquad Q_R\succeq0,\quad\lambda_{R,\ell}\ge0,\quad m=[1;\zeta].
$$

All multipliers are independent nonnegative constants. Local transition
residuals have three variables $(\omega_n,\xi_n)$; global residuals have $2N$.
The global good union is enforced branchwise on each individual good disk;
the global bad intersection uses all local bad generators. Enforcing bad
residuals also on the closed exterior is a conservative boundary strengthening.

| Problem | Unknowns for a fixed controller | Objective and structure |
| --- | --- | --- |
| Monolithic | Global free quadratic $B_q,V_q$, Gram matrices, constant SOS multipliers | $\min0$; affine coefficient identities and PSD/nonnegative cones; MOSEK SDP. |
| Compositional | Separate local $B_{n,q},V_{n,q}$, both supplies per converter, local Grams/multipliers and clique matrices | $\min0$; local SOS constraints and both aggregate PSD matching systems in one centrally assembled MOSEK SDP. |

The compositional constraints are sufficient restrictions, so their feasible
set need not equal the monolithic SDP's feasible set. Degree, plant, controller
family, labels and progress units are matched. There is no ADMM iteration,
TSSOS sparsity heuristic, sampled replacement for SOS, or CEGIS stage.

### Exact clique representation

For each supply family $S$, the implementation matches every entry of

$$
-\sum_nT_n^\top S_nT_n=\sum_C E_C^\top Z_CE_C,\qquad Z_C\succeq0.
$$

The chordal extension eliminates current coordinates before voltage
coordinates. Fill entries are matched too, including entries that must cancel
to zero. Checking only clique principal submatrices would not establish the
original aggregate inequality.

### Stationary-face reduction

For the implemented $0<\alpha<1$, a stationary self-transition with zero
required decrease and zero centered supply forces both barrier and rank
residuals to vanish at the origin. If all its domain generators are strictly
positive there, the nonnegative constant multipliers must be zero and the
PSD Gram's constant row/column must vanish. The implementation encodes this
implied face exactly, including odd self-transitions allocated zero progress.
It adds neither Lyapunov stability nor a positive robustness margin.

## 7. Exact verification

The numerical SDP uses floating-point coefficients. Certification uses a
reconstructed **rational witness**:

1. Preserve the raw solver iterate.
2. Round candidate free entries on the established dyadic grids and solve the
   affine coefficient identities exactly, eliminating auxiliary Gram/clique
   entries first.
3. Require every identity to vanish, every multiplier to be nonnegative and
   every Gram/clique matrix to pass an exact rational PSD check.
4. Reconstruct and check both original undecomposed aggregate matrices.
5. Export physical-coordinate polynomials and complete SOS witnesses.
6. Rebuild and verify the saved result in a fresh Julia process before the
   supervisor reports `CERTIFIED`.

The saved-controller verifier also evaluates the original physical equations
exactly at all four vertices of each state box for `0 <= u_n <= 1`, and all
eight vertices of the state/internal-input box for successor containment.
The internal input is nominal $w_n^0\in[-1,1]$, not centered $\omega_n$.
Affineness makes these vertex checks sufficient on the full boxes.

Repair retains the original order: grids `32,42,52` bits and thresholds
`1e-9,1e-7,0`. A deliberately changed coefficient must fail the exact checker.
Rational LDL tests handle semidefinite boundary cases without introducing a
positive numerical margin. Floating-point eigenvalues are diagnostics only.

## 8. Hyperparameters and comparison results

| Setting | Value |
| --- | --- |
| Sizes | 2, 10, 50, 100 by default |
| Seed / repetitions | 20261122 / one per campaign |
| Controller budget | 3 candidates in the launcher |
| Degree / Gram monomial degree / multiplier degree | 2 / 1 / 0 |
| Barrier factor | $\alpha=0.01$ |
| Global decrease | $0.001N$ |
| Progress policy | `auto` |
| Solver / BLAS threads | One each |
| Requested primal/dual feasibility tolerance | $10^{-9}$ |
| Timed-search / whole-worker guards | 600 s / 900 s |
| Per-SOS and per-controller-LP solver limit | 120 s |
| Sampled process-group RSS guard | 8 GiB |


Both methods use the same generated plant at every network size. The
compositional implementation is a centrally assembled SDP with separate local
certificates and exact clique coupling. The monolithic implementation uses
unrestricted dense global quadratic certificates of the same degree.

| N | Compositional search (s) | Monolithic search (s) | Monolithic outcome |
| --- | ---: | ---: | --- |
| 2 | 0.392 | 0.361 | Certified |
| 10 | 1.66 | 12.86 | Certified |
| 50 | 14.55 | >=600 | SOS construction timeout |
| 100 | 88.43 | >=600 | SOS construction timeout |

All four compositional runs and both completed monolithic runs returned
`OPTIMAL`, passed exact verification and passed fresh-process reload. Larger
monolithic runs stopped during symbolic SOS construction, before MOSEK
optimization. These are censored resource outcomes, not infeasibility results.
One seed and one repetition do not establish statistical or generic runtime
claims.

The [comparison CSV](Reproducibility/comparison.csv) retains all eight original
measurements. Refactored execution can change overhead; post-run validation
timings are not substituted for those measurements.

Search totals include all controller attempts, physical reconstruction, SOS
construction/solution, exact proof construction and orchestration. Explicit
startup, unrelated warmup, file export and fresh reload are separate.
Rejected-attempt time is a subset of the other phase totals, not an additional
charge. Partial construction times at a guard interruption are not completed
solve times.

### Published data

All retained data are consolidated in [Reproducibility](Reproducibility/).

| Material | Purpose |
| --- | --- |
| [comparison.csv](Reproducibility/comparison.csv) | Eight timing rows, resource outcomes, SDP sizes and plant fingerprints. |
| [config.json](Reproducibility/config.json) | Original seed, search settings, resource limits and timed-source hashes. |
| [hardware.json](Reproducibility/hardware.json) | Computer specifications for interpreting the timing measurements. |
| [status.json](Reproducibility/status.json) | Completion status of the reported campaign. |
| `parameter_draws_N*.toml` | Exact physical parameters, line conductances, signal matrix, domains and plant fingerprint at each size. |
| [controller_search.json](Reproducibility/controller_search.json) | Original controller attempts, exact coefficients, progress allocations and software versions for all eight runs. |
| `N*_compositional/` and `N*_monolithic/` | Per-method timing/resource record and worker log; each successful run also contains its complete exact `result.toml`. |
| [verification_summary.toml](Reproducibility/verification_summary.toml) | Matched-input, physical-vertex and full saved-witness verification results. |
| [original_timed_sources.zip](Reproducibility/original_timed_sources.zip) | Unchanged source files from the timed campaign, matching the hashes in `config.json`. |

There are six saved witnesses: compositional results for all four sizes and
monolithic results for `N=2,10`. The two larger monolithic runs stopped during
construction and retain their timeout records and logs.

Each archived size has $N$ distinct physical parameter tuples. Candidate order,
equilibrium screening and recorded first candidates/progress allocations match
between methods. The exact post-run checks passed at 696 input vertices and
1392 successor vertices across the six exported controllers. Individual vertex
values can be recomputed with `verify_controller_vertices(base, theta)`.

New campaigns create their own `Results/` directories by default. Generated
campaigns are excluded from Git; the published measurements remain in
`Reproducibility/`.

To regenerate the CSV and a Markdown comparison from the retained run records:

```bash
python3 Compare_Direct_CPC.py summarize Reproducibility
```

## 9. Verifying a saved certificate

A result contains the physical controller parameters, free polynomial
coefficients, both supply families, SOS witnesses, raw solver coefficients and
repaired exact coefficients. For example, rebuild the saved `N=100`
compositional witness with:

```bash
julia --project=. Benchmark_Direct_CPC.jl verify \
  Reproducibility/N100_compositional/result.toml Results/Reload_N100
```

Substitute another successful method/size to check its saved witness. The
verifier reconstructs the exact model and every coefficient identity, checks
all rational PSD/nonnegative conditions, checks both original aggregate supply
matrices, compares the exported physical-coordinate polynomials, and verifies
the original physical input/successor vertex bounds using nominal $w_n^0$.
It rebuilds the saved proof without rerunning optimization. The output records
success or failure; solver status and rounded coefficients alone are not
accepted as certification.

## 10. Displaying B and V as readable polynomials

The certificate values are in each successful run's `result.toml`, under
`result["polynomials"]`. The `controller_search.json` file records the controller
search; the complete polynomial coefficients belong to the saved witness.

For a monolithic result, the polynomial names are `B_bad`, `B_good`, `V_bad`
and `V_good`. For a compositional result, they are `B_n_bad`, `B_n_good`,
`V_n_bad` and `V_n_good`, where `n` is the converter index. All `V` entries
in this two-priority example represent the rank for odd priority 1, evaluated
at the indicated shared automaton state.

The `show` command expands the stored coefficient matrices into ordinary
polynomial expressions. For example:

```bash
julia --project=. Benchmark_Direct_CPC.jl show \
  Reproducibility/N2_monolithic/result.toml

julia --project=. Benchmark_Direct_CPC.jl show \
  Reproducibility/N2_compositional/result.toml
```

The default is **physical coordinates**, with coefficients displayed to
10 significant digits. The complete syntax is:

```text
show RESULT_TOML [physical|normalized|centered] [decimal|exact] [SUBSYSTEM|all]
```

| Option | Meaning |
| --- | --- |
| `physical` | Original physical variables `i_n`, `v_n`; uses `original_physical_coefficient_matrix_exact`. |
| `normalized` | Nominal normalized variables `z_n_1`, `z_n_2`; uses `original_normalized_coefficient_matrix_exact`. |
| `centered` | Variables `xi_n = z_n - e_n`; uses `centered_coefficient_matrix_exact` and prints the saved equilibrium shift in the selected numeric format. |
| `decimal` | Rounded display of the repaired exact coefficients, for inspection. |
| `exact` | Exact rational coefficients with no display rounding. |
| `SUBSYSTEM` | A converter index for compositional results; `all` displays all converters. Monolithic polynomials use the full state. |

Inspect just converter 3 of the `N=100` compositional result in compact,
centered coordinates:

```bash
julia --project=. Benchmark_Direct_CPC.jl show \
  Reproducibility/N100_compositional/result.toml centered decimal 3
```

Display its original physical-coordinate coefficients exactly:

```bash
julia --project=. Benchmark_Direct_CPC.jl show \
  Reproducibility/N100_compositional/result.toml physical exact 3
```

The underlying matrix convention is

$$
p(s)=\begin{bmatrix}1\\s\end{bmatrix}^{\!T}
H\begin{bmatrix}1\\s\end{bmatrix}.
$$

Thus linear and mixed quadratic coefficients are **twice** the corresponding
off-diagonal entries of $H$. The viewer performs this expansion automatically.
For monolithic certificates, the variable order is
$(i_1,v_1,\ldots,i_N,v_N)$, or its normalized/centered counterpart; a local
polynomial uses only its converter's two coordinates.

The viewer prints the stored, unscaled $B,V$ and the saved global progress
step $\varepsilon$. To obtain a unit-decrease global certificate, divide
monolithic $B,V$, or the sums of local $B_n,V_n$, by this global step.
Supply matrices are stored separately under `result["supplies_exact"]`.

`show` reads the saved witness without solving or modifying anything. Decimal
expressions are approximate displays; use `verify` and the stored exact
rational coefficients for certification. The Julia function
`print_saved_certificates(io, path; coordinates=:physical, exact=false,
subsystem=nothing)` also provides this view from an interactive session.
