"""Convex cost model, gradients, and feasible-set projections."""

import numpy as np


def _rates_by_scenario(productivity, num_scenarios):
    """Return productivity with shape [scenario, process, 1]."""
    rates = np.asarray(productivity, dtype=float)
    if rates.ndim == 1:
        rates = np.tile(rates, (num_scenarios, 1))
    if rates.shape[0] != num_scenarios:
        raise ValueError("Productivity rows must match the number of scenarios.")
    return rates[:, :, None]


def common_cost(x, data):
    """Labor plus smooth hour-to-hour reassignment cost."""
    x = np.asarray(x, dtype=float)
    labor = np.sum(data["labor_cost"] * x)
    changes = x[:, 1:] - x[:, :-1]
    adjustment = data["adjustment_penalty"] * np.sum(changes**2)
    return float(labor + adjustment)


def common_gradient(x, data):
    """Gradient of labor and adjustment costs, shape [process, period]."""
    x = np.asarray(x, dtype=float)
    gradient = data["labor_cost"].copy()
    differences = x[:, 1:] - x[:, :-1]
    # Each squared difference contributes with opposite signs to its two hours.
    gradient[:, :-1] -= 2.0 * data["adjustment_penalty"] * differences
    gradient[:, 1:] += 2.0 * data["adjustment_penalty"] * differences
    return gradient


def scenario_losses(x, workloads, data, productivity=None):
    """Return F_s(x) for every scenario as a one-dimensional array."""
    workloads = np.asarray(workloads, dtype=float)
    rates = _rates_by_scenario(
        data["productivity"] if productivity is None else productivity,
        workloads.shape[0],
    )
    capacity = rates * np.asarray(x, dtype=float)[None, :, :]
    shortage = np.maximum(workloads - capacity, 0.0)
    weighted_square = np.sum(
        data["shortage_weights"][None, :, None] * shortage**2, axis=(1, 2)
    )
    return common_cost(x, data) + data["shortage_penalty"] * weighted_square


def scenario_gradients(x, workloads, data, productivity=None):
    """Return all analytic gradients with shape [scenario, process, period].

    For q=[d-rx]_+, d(q^2)/dx = -2*r*q when q>0 and zero when
    q=0. The derivative is continuous at the positive-part kink.
    """
    workloads = np.asarray(workloads, dtype=float)
    rates = _rates_by_scenario(
        data["productivity"] if productivity is None else productivity,
        workloads.shape[0],
    )
    capacity = rates * np.asarray(x, dtype=float)[None, :, :]
    shortage = np.maximum(workloads - capacity, 0.0)
    shortage_gradient = (
        -2.0
        * data["shortage_penalty"]
        * data["shortage_weights"][None, :, None]
        * rates
        * shortage
    )
    return common_gradient(x, data)[None, :, :] + shortage_gradient


def project_capped_simplex(v, upper, budget):
    """Euclidean projection onto {0 <= x <= upper, sum(x) <= budget}.

    If clipping to the box already satisfies the budget, it is the projection.
    Otherwise the KKT conditions give x_i=clip(v_i-theta, 0, upper_i).
    Bisection finds the unique threshold theta whose projected entries sum to
    the available workforce.
    """
    v = np.asarray(v, dtype=float)
    upper = np.asarray(upper, dtype=float)
    clipped = np.clip(v, 0.0, upper)
    if clipped.sum() <= budget:
        return clipped

    low = float(np.min(v - upper))
    high = float(np.max(v))
    for _ in range(80):
        theta = 0.5 * (low + high)
        candidate = np.clip(v - theta, 0.0, upper)
        if candidate.sum() > budget:
            low = theta
        else:
            high = theta
    return np.clip(v - high, 0.0, upper)


def project_staffing(x, data):
    """Project a [process, period] plan independently in each hour."""
    x = np.asarray(x, dtype=float)
    projected = np.empty_like(x)
    for t in range(data["num_periods"]):
        projected[:, t] = project_capped_simplex(
            x[:, t], data["max_staffing"][:, t], data["workforce_available"][t]
        )
    return projected


def project_simplex(p):
    """Euclidean projection onto {p >= 0, sum(p)=1}."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p must be a nonempty one-dimensional vector.")
    sorted_p = np.sort(p)[::-1]
    cumulative = np.cumsum(sorted_p)
    indices = np.arange(1, p.size + 1)
    active = sorted_p - (cumulative - 1.0) / indices > 0.0
    rho = np.nonzero(active)[0][-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1.0)
    projected = np.maximum(p - theta, 0.0)
    return projected / projected.sum()


def saddle_operator(y, p, data):
    """Scaled saddle operator G(y,p)=[grad_y L; -grad_p L].

    The physical staffing plan is x=worker_scale*y. The full saddle objective
    is divided by objective_scale solely to keep the primal and dual numerical
    magnitudes convenient for one extragradient step size.
    """
    x = data["worker_scale"] * np.asarray(y, dtype=float)
    losses = scenario_losses(x, data["train_workloads"], data)
    gradients_x = scenario_gradients(x, data["train_workloads"], data)
    grad_y = (
        data["worker_scale"] / data["objective_scale"]
    ) * np.tensordot(p, gradients_x, axes=(0, 0))
    operator_p = -losses / data["objective_scale"]
    return grad_y, operator_p


def project_scaled_staffing(y, data):
    """Project normalized variables by mapping to workers and back."""
    return project_staffing(data["worker_scale"] * y, data) / data["worker_scale"]


def natural_residual(y, p, data):
    """Natural VI residual ||z-P(z-G(z))|| in normalized coordinates."""
    grad_y, operator_p = saddle_operator(y, p, data)
    y_projected = project_scaled_staffing(y - grad_y, data)
    p_projected = project_simplex(p - operator_p)
    return float(
        np.sqrt(np.sum((y - y_projected) ** 2) + np.sum((p - p_projected) ** 2))
    )


def finite_difference_gradient_check(data, checks=12, epsilon=1e-5, seed=41):
    """Check several random coordinates for several training scenarios."""
    rng = np.random.default_rng(seed)
    x = project_staffing(0.82 * data["forecast"] / data["productivity"][:, None], data)
    max_absolute_error = 0.0
    max_relative_error = 0.0

    for _ in range(checks):
        s = int(rng.integers(data["num_train_scenarios"]))
        j = int(rng.integers(data["num_processes"]))
        t = int(rng.integers(data["num_periods"]))
        analytic = scenario_gradients(
            x, data["train_workloads"][s : s + 1], data
        )[0, j, t]
        direction = np.zeros_like(x)
        direction[j, t] = epsilon
        plus = scenario_losses(
            x + direction, data["train_workloads"][s : s + 1], data
        )[0]
        minus = scenario_losses(
            x - direction, data["train_workloads"][s : s + 1], data
        )[0]
        numerical = (plus - minus) / (2.0 * epsilon)
        absolute_error = abs(analytic - numerical)
        relative_error = absolute_error / max(1.0, abs(analytic), abs(numerical))
        max_absolute_error = max(max_absolute_error, absolute_error)
        max_relative_error = max(max_relative_error, relative_error)

    return {
        "coordinates_checked": checks,
        "max_absolute_error": float(max_absolute_error),
        "max_relative_error": float(max_relative_error),
        "passed": bool(max_relative_error < 2e-6),
    }


def check_staffing_feasibility(x, data, tolerance=1e-8):
    """Return feasibility diagnostics for a physical staffing plan."""
    x = np.asarray(x)
    return {
        "minimum_staffing": float(x.min()),
        "maximum_box_violation": float(np.maximum(x - data["max_staffing"], 0).max()),
        "maximum_workforce_violation": float(
            np.maximum(x.sum(axis=0) - data["workforce_available"], 0).max()
        ),
        "feasible": bool(
            x.min() >= -tolerance
            and np.all(x <= data["max_staffing"] + tolerance)
            and np.all(x.sum(axis=0) <= data["workforce_available"] + tolerance)
        ),
    }

