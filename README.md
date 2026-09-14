# Robust Fulfillment-Center Labor Allocation under Workload Uncertainty

This compact Operations Research project compares nominal, expected-value, and
finite-scenario robust staffing plans for a synthetic fulfillment center. The
planning decision is how many workers to assign to Stow, Pick, Pack, and Ship in
each of 12 one-hour periods.

> **This project uses synthetic data and is not based on Amazon proprietary data
> or internal systems.** It is a simplified fulfillment-center-style model built
> for demonstration and learning, not an exact representation of Amazon operations.

## Business motivation

Intraday labor must be assigned before workload is fully known. A lean plan saves
labor but can create backlog when volume arrives above forecast or productivity
falls. A protective plan improves service resilience but may cost more and leave
some processing capacity unused. The experiment makes that trade-off measurable.

The synthetic instance has four processes, 12 hours, 60 training scenarios, and
six independent stress regimes with 200 scenarios each. Workload uncertainty
combines building-wide, process-specific, time-correlated, and localized shocks.

## Mathematical model

Let $x_{jt}$ be workers assigned to process $j$ in hour $t$, $d^{(s)}_{jt}$
be required units in scenario $s$, and $r_j$ be units processed by one worker
in one hour. Scenario loss is

$$
F_s(x)=C_{\mathrm{labor}}(x)+C_{\mathrm{adjust}}(x)
       +C_{\mathrm{shortage}}^{(s)}(x),
$$

where

$$
C_{\mathrm{labor}}(x)=\sum_{j,t}c_{jt}x_{jt},
$$

$$
C_{\mathrm{adjust}}(x)=\gamma\sum_j\sum_{t=2}^{T}
                    (x_{jt}-x_{j,t-1})^2,
$$

and

$$
C_{\mathrm{shortage}}^{(s)}(x)=\lambda\sum_{j,t}w_j
              [d^{(s)}_{jt}-r_jx_{jt}]_+^2.
$$

The squared positive-part penalty is convex and continuously differentiable.
Its staffing derivative is

$$
-2\lambda w_jr_j[d^{(s)}_{jt}-r_jx_{jt}]_+.
$$

The feasible set is

$$
0\le x_{jt}\le \bar x_{jt},\qquad
\sum_jx_{jt}\le W_t\quad\text{for every hour }t.
$$

The adjustment term represents reassignment friction without adding hard ramp
constraints to the projection. Projection onto the feasible set separates by
hour. For a tentative vector $v$, the code finds

$$
x_j=\mathrm{clip}(v_j-\theta,0,\bar x_j)
$$

with a bisection search for the threshold $\theta$ when the workforce cap binds.
This is the exact Euclidean projection onto the box-constrained workforce budget.

The default synthetic parameters in `data.py` are $\lambda=0.0015$,
$\gamma=1.25$, process shortage weights $(0.85,1.20,1.15,1.35)$, and
productivity rates $(105,92,78,118)$ units per worker-hour for Stow, Pick,
Pack, and Ship. Process staffing limits are $(58,72,78,66)$ workers. These
values are illustrative calibrations, not estimates from operational data.

## Three planning methods

The methods share the same data dictionary and return result dictionaries with a
staffing array under `result["x"]`.

**Nominal planner**

$$
\min_{x\in X}F_{\mathrm{forecast}}(x).
$$

**Expected-value planner**

$$
\min_{x\in X}\frac{1}{S}\sum_{s=1}^{S}F_s(x).
$$

Both smooth convex baselines are solved with SciPy SLSQP and analytic gradients.

**Robust planner**

$$
\min_{x\in X}\max_s F_s(x)
=\min_{x\in X}\max_{p\in\Delta_S}\sum_s p_sF_s(x),
$$

where $\Delta_S=\{p\ge0:\mathbf 1^\top p=1\}$. For
$L(x,p)=\sum_sp_sF_s(x)$, the saddle operator is

$$
G(x,p)= \begin{bmatrix}
\nabla_x L(x,p)\\
-\nabla_p L(x,p)
\end{bmatrix}
=
\begin{bmatrix}
\sum_sp_s\nabla F_s(x)\\
-(F_1(x),\ldots,F_S(x))
\end{bmatrix}.
$$

Because the second block is negative loss, a projected descent step on this
operator is an ascent step in $p$. The code applies Korpelevich's two-stage
projected extragradient method:

$$
z_{k+1/2}=P(z_k-\eta G(z_k)),\qquad
z_{k+1}=P(z_k-\eta G(z_{k+1/2})).
$$

The primal projection is the hourly capped-simplex projection described above;
the dual projection uses the standard sorting-and-thresholding Euclidean
projection onto the probability simplex. Worker variables and objective values are
rescaled internally for numerical conditioning, which does not change the model
or its solution.

Convergence history records the worst training loss, normalized primal and dual
iterate changes, and the natural variational-inequality residual

$$
\|z-P(z-G(z))\|.
$$

An independent SciPy epigraph solve is included as a numerical reference:

$$
\min_{x,q}\ q\quad\text{s.t.}\quad F_s(x)\le q\quad\forall s.
$$

It is used for verification, not as the reported robust policy. The implementation
internally divides the epigraph variable and constraints by `objective_scale=10000`
to improve SLSQP conditioning; this positive scaling leaves the optimizer unchanged.

## Synthetic experiment and stress tests

The point forecast has an early Stow peak and later Pick, Pack, and Ship peaks.
Normal productivity ranges from 78 to 118 units per worker-hour. Fixed random
seeds make both training and held-out scenarios reproducible.

Plans are tested independently under:

1. normal workload variability;
2. a network-wide volume surge (a facility-wide volume increase in this model);
3. a Pick-heavy spike;
4. a late-day demand surge;
5. reduced worker productivity; and
6. a combined volume surge and productivity shock.

The last two regimes change the productivity used in evaluation while leaving the
planned assignments fixed. No test scenario is used for optimization.

## Business metrics

For each plan and evaluation set, the project reports:

- staffing labor cost and total allocated labor-hours;
- the quadratic staffing-adjustment cost;
- average and worst total shortage;
- maximum shortage in any single hour, aggregated across processes;
- average and worst total model loss;
- workload-weighted service level,
  $1-\text{total unmet units}/\text{total required units}$; and
- utilization, useful processed units divided by supplied processing capacity.

Utilization may fall when a planner carries protective capacity; this is the
intended efficiency-versus-resilience trade-off.

`service_level` and `utilization` are stored in the CSV files as proportions in
$[0,1]$. The plotting code multiplies service level by 100 for percentage display.
Total model loss combines labor cost, adjustment cost, and the calibrated squared
shortage penalty. It should be interpreted as a synthetic cost-equivalent model
objective, not as an audited accounting amount.

## Actual results from the default run

The following numbers were generated by `python main.py` with seed 2026 and match
`results/overall_stress_summary.csv`. The table pools 1,200 independent scenarios
across all six stress regimes. Because every regime contains 200 scenarios, this
is an equal-weighted stress suite, not an estimate based on real-world regime
probabilities.

| Planner | Labor cost ($) | Labor-hours | Avg. shortage | Worst shortage | Service level | Utilization | Avg. total model loss | Worst total model loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 47,109.85 | 1,819.17 | 29,124.25 | 106,241.93 | 84.92% | 99.04% | 113,927.98 | 561,232.67 |
| Expected | 51,633.39 | 1,993.45 | 17,774.27 | 91,925.45 | 90.80% | 96.66% | 85,843.33 | 432,061.78 |
| Robust | 53,990.65 | 2,083.46 | 13,942.16 | 85,035.17 | 92.78% | 94.79% | 79,957.32 | 386,070.01 |

Relative to expected-value planning, robustness requires **4.57% more labor
cost** and **4.52% more labor-hours**. Across the pooled stresses it reduces
average shortage by **21.56%**, worst observed shortage by **7.50%**, and worst
total loss by **10.64%**, while increasing service level by **1.98 percentage
points**. It also reduces average model loss across this equal-weighted stress
suite by **6.86%**. Relative to the nominal plan, the robust plan reduces worst
total loss by **31.21%**.

Labor cost, labor-hours, and adjustment cost are plan-level quantities, so they
are identical for a given planner in every row of `results/summary.csv`. Workload,
productivity, shortage, service, utilization, and total loss vary by regime.

### In-sample training scenarios

These rows come from `results/summary.csv` where `sample` is `In-sample` and
`regime` is `Training scenarios`.

| Planner | Avg. shortage | Worst shortage | Service level | Utilization | Avg. total model loss | Worst total model loss |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 11,940.73 | 44,469.77 | 93.45% | 98.46% | 61,759.21 | 129,820.90 |
| Expected | 3,942.80 | 27,932.10 | 97.84% | 94.09% | 56,201.42 | 85,865.14 |
| Robust | 2,488.00 | 20,235.18 | 98.64% | 91.03% | 57,482.36 | 73,383.23 |

Expected-value planning has the lowest average training loss, as its objective
requires. Robust planning accepts a **2.28% higher** average training loss than
Expected while reducing worst training loss by **14.54%**. This is the intended
average-performance versus tail-protection trade-off.

### Independent stress-regime service levels

The values below are the `Out-of-sample` rows of `results/summary.csv`. The CSV
stores proportions; the table and `stress_test_service_level.png` display those
values after multiplication by 100.

| Regime | Nominal | Expected | Robust |
|---|---:|---:|---:|
| Normal variability | 95.35% | 98.90% | 99.30% |
| Network-wide volume surge | 83.11% | 90.40% | 93.15% |
| Pick-heavy spike | 91.72% | 95.90% | 97.13% |
| Late-day demand surge | 87.15% | 93.24% | 94.82% |
| Reduced worker productivity | 85.41% | 92.57% | 95.01% |
| Combined surge + productivity shock | 69.67% | 76.33% | 79.51% |

The difference is visible in specific operational shocks:

| Regime, robust vs. expected | Avg. shortage reduction | Worst-loss reduction | Service-level gain |
|---|---:|---:|---:|
| Network-wide volume surge | 28.64% | 14.52% | 2.75 pp |
| Reduced worker productivity | 32.88% | 14.02% | 2.44 pp |
| Combined surge + productivity shock | 13.44% | 10.64% | 3.18 pp |

Under normal held-out variability, service levels are 95.35%, 98.90%, and 99.30%
for nominal, expected, and robust plans, respectively. Under the deliberately
severe combined shock they fall to 69.67%, 76.33%, and 79.51%. The absolute
levels should not be treated as operational targets; they show how the same fixed
plans degrade under increasingly adverse synthetic conditions.

Under normal variability, Expected has the lowest average model loss (53,990.91),
while Robust has a 4.18% higher average loss (56,246.73) but a 7.88% lower worst
loss. Under the combined shock, utilization is approximately 100% for every plan
while service remains below 80%, indicating capacity saturation rather than good
operational efficiency.

The robust method converged in 250 iterations to a natural residual of
$1.40\times10^{-5}$. Its worst training loss was 73,383.227964 model-loss units,
versus 73,383.225044 for the independent SciPy epigraph reference. The absolute
difference was approximately 0.00292 and the relative gap was
$3.98\times10^{-8}$. Nominal, Expected, Robust, and the SciPy reference all
reported `converged`. Twelve randomly selected analytic gradient coordinates
matched central finite differences with maximum absolute error
$1.20\times10^{-6}$ and maximum relative error $4.75\times10^{-9}$.

These results demonstrate a modest, interpretable insurance premium: robust
staffing sacrifices some utilization and labor cost to improve tail protection.
They do not establish that robust planning will dominate on every distribution,
sample, or business metric.

## Figures and result files

Running the project creates:

- `figures/workload_forecast.png` — synthetic process workload by hour;
- `figures/staffing_allocation.png` — one subplot per process comparing all three planners and the process staffing limit;
- `figures/stress_test_service_level.png` — service by regime and planner;
- `figures/robust_convergence.png` — worst loss and natural residual;
- `results/summary.csv` — every planner-by-regime metric;
- `results/overall_stress_summary.csv` — pooled independent results; and
- `results/solver_validation.csv` — gradient and robust-solver checks.

## Project structure

```text
robust-fc-labor-planning/
├── README.md
├── requirements.txt
├── main.py
├── data.py
├── model.py
├── solvers.py
├── experiments.py
├── plotting.py
├── figures/
│   ├── workload_forecast.png
│   ├── staffing_allocation.png
│   ├── stress_test_service_level.png
│   └── robust_convergence.png
└── results/
    ├── summary.csv
    ├── overall_stress_summary.csv
    └── solver_validation.csv
```

## Installation and use

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install the four declared dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete experiment from the project directory:

```bash
python main.py
```

`main.py` creates output folders automatically, checks gradients and projections,
solves all three planners, checks feasibility and finite values, runs independent
stress tests, writes the result tables, creates the figures, and validates the
robust objective against SciPy.

The deliberately plain APIs also support direct experimentation:

```python
from data import generate_data
from solvers import solve_nominal, solve_expected, solve_robust_extragradient

data = generate_data(seed=2026)
nominal = solve_nominal(data)
expected = solve_expected(data)
robust = solve_robust_extragradient(data, step_size=0.035)
```

## Limitations and extensions

This is a continuous, single-building model. Workers can be fractional, workflow
coupling is represented only through common labor availability, and backlog does
not explicitly carry between hours. Productivity and costs are simplified, and
the robust guarantee applies only to the finite training scenario set.

Natural extensions include integer shifts, skill compatibility, breaks and labor
law constraints, cross-training matrices, explicit backlog flow, downstream
process coupling, attendance uncertainty, rolling-horizon re-optimization, CVaR
or distributionally robust objectives, and calibration from appropriately
authorized operational data.

## Skills Demonstrated

- Operations Research and convex optimization
- robust optimization and saddle-point reformulation
- projected extragradient algorithms and simplex projection
- labor allocation and capacity-risk management
- scenario generation, simulation, and out-of-sample stress testing
- analytic gradients and numerical verification
- Python, NumPy, SciPy, Pandas, and Matplotlib
