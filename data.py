"""Synthetic fulfillment-center workload data.

All arrays use the axis order [process, period], or
[scenario, process, period] when a scenario axis is present.
"""

import numpy as np


PROCESS_NAMES = ["Stow", "Pick", "Pack", "Ship"]


def _base_workload():
    """Return a transparent 4-by-12 point forecast in units per hour."""
    return np.array(
        [
            [3600, 4100, 4500, 4300, 3900, 3500, 3200, 3000, 2800, 2700, 2500, 2300],
            [2100, 2400, 2800, 3300, 3900, 4500, 5000, 5400, 5700, 5500, 4900, 4100],
            [1700, 1900, 2200, 2600, 3100, 3700, 4300, 4900, 5400, 5700, 5500, 4900],
            [1200, 1400, 1600, 1900, 2300, 2800, 3400, 4100, 4800, 5500, 6000, 6100],
        ],
        dtype=float,
    )


def _sample_workloads(forecast, num_scenarios, rng, include_tail_events=True):
    """Generate correlated workload scenarios around a point forecast.

    Noise has four interpretable pieces: a building-wide volume factor,
    process-specific variation, smooth hour-to-hour variation, and occasional
    localized surges. Multipliers are clipped only to avoid implausibly small
    synthetic workloads.
    """
    num_processes, num_periods = forecast.shape
    common = rng.normal(0.0, 0.055, size=(num_scenarios, 1, 1))
    process = rng.normal(0.0, 0.045, size=(num_scenarios, num_processes, 1))

    raw_time = rng.normal(0.0, 0.045, size=(num_scenarios, num_periods))
    # Neighboring hours are correlated: workload forecast errors rarely jump
    # independently from one hour to the next.
    smooth_time = (
        0.25 * np.roll(raw_time, 1, axis=1)
        + 0.50 * raw_time
        + 0.25 * np.roll(raw_time, -1, axis=1)
    )[:, None, :]
    local = rng.normal(0.0, 0.025, size=(num_scenarios, num_processes, num_periods))

    multiplier = 1.0 + common + process + smooth_time + local

    if include_tail_events:
        for s in range(num_scenarios):
            event = rng.random()
            if event < 0.10:  # building-wide unplanned volume
                multiplier[s] += rng.uniform(0.10, 0.18)
            elif event < 0.19:  # Pick/Pack concentration
                multiplier[s, 1:3, 5:10] += rng.uniform(0.12, 0.22)
            elif event < 0.27:  # late wave release
                multiplier[s, :, 8:12] += rng.uniform(0.12, 0.20)

    multiplier = np.clip(multiplier, 0.72, None)
    return forecast[None, :, :] * multiplier


def generate_data(num_train_scenarios=60, seed=2026):
    """Return every input needed by the three planners in one dictionary."""
    rng = np.random.default_rng(seed)
    forecast = _base_workload()
    train_workloads = _sample_workloads(
        forecast, num_train_scenarios, rng, include_tail_events=True
    )

    # Productivity is synthetic processed units per worker-hour.
    productivity = np.array([105.0, 92.0, 78.0, 118.0])
    labor_cost = np.array([25.0, 27.0, 26.0, 25.0])[:, None]
    labor_cost = labor_cost * np.ones((1, forecast.shape[1]))

    # Available associates vary modestly through the shift. The constraint is
    # shared across processes in each hour.
    workforce_available = np.array(
        [108, 118, 130, 145, 162, 178, 198, 216, 226, 232, 226, 210], dtype=float
    )
    max_staffing = np.array([58.0, 72.0, 78.0, 66.0])[:, None]
    max_staffing = max_staffing * np.ones((1, forecast.shape[1]))

    return {
        "process_names": PROCESS_NAMES.copy(),
        "period_labels": [f"Hour {t + 1}" for t in range(forecast.shape[1])],
        "num_processes": forecast.shape[0],
        "num_periods": forecast.shape[1],
        "num_train_scenarios": num_train_scenarios,
        "forecast": forecast,
        "train_workloads": train_workloads,
        "productivity": productivity,
        "labor_cost": labor_cost,
        "workforce_available": workforce_available,
        "max_staffing": max_staffing,
        "shortage_penalty": 0.0015,
        "shortage_weights": np.array([0.85, 1.20, 1.15, 1.35]),
        "adjustment_penalty": 1.25,
        # The robust algorithm works in y=x/worker_scale and scales dollar
        # losses. Scaling changes neither the feasible x nor the saddle point.
        "worker_scale": 50.0,
        "objective_scale": 10000.0,
        "seed": seed,
    }


def generate_stress_tests(data, scenarios_per_regime=200, seed=90210):
    """Create independent, named stress regimes for held-out evaluation.

    Each dictionary entry contains [scenario, process, period] workloads and
    [scenario, process] productivity rates. Productivity therefore can change
    by regime without changing the plan computed under normal assumptions.
    """
    forecast = data["forecast"]
    base_rates = data["productivity"]
    regimes = {}

    # Separate deterministic seeds make each regime reproducible and make it
    # easy to change one regime without perturbing every other regime.
    for offset, name in enumerate(
        [
            "Normal variability",
            "Network-wide volume surge",
            "Pick-heavy spike",
            "Late-day demand surge",
            "Reduced worker productivity",
            "Combined surge + productivity shock",
        ]
    ):
        rng = np.random.default_rng(seed + offset)
        workloads = _sample_workloads(
            forecast, scenarios_per_regime, rng, include_tail_events=False
        )
        rates = np.tile(base_rates, (scenarios_per_regime, 1))

        if name == "Network-wide volume surge":
            workloads *= rng.normal(1.18, 0.025, size=(scenarios_per_regime, 1, 1))
        elif name == "Pick-heavy spike":
            workloads[:, 1, 5:10] *= rng.normal(
                1.32, 0.035, size=(scenarios_per_regime, 1)
            )
        elif name == "Late-day demand surge":
            workloads[:, :, 8:12] *= rng.normal(
                1.25, 0.030, size=(scenarios_per_regime, 1, 1)
            )
        elif name == "Reduced worker productivity":
            rates *= rng.normal(0.88, 0.012, size=(scenarios_per_regime, 1))
        elif name == "Combined surge + productivity shock":
            workloads *= rng.normal(1.16, 0.025, size=(scenarios_per_regime, 1, 1))
            workloads[:, :, 8:12] *= 1.08
            rates *= rng.normal(0.86, 0.012, size=(scenarios_per_regime, 1))

        regimes[name] = {"workloads": workloads, "productivity": rates}

    return regimes
