"""Three planners with deliberately simple, interchangeable function APIs."""

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from model import (
    natural_residual,
    project_scaled_staffing,
    project_simplex,
    project_staffing,
    saddle_operator,
    scenario_gradients,
    scenario_losses,
)


def _scipy_constraints(data):
    """Build bounds and per-hour workforce constraints for flattened x."""
    size = data["num_processes"] * data["num_periods"]
    # C-order flattening stores every period for process 0, then process 1.
    workforce_matrix = np.zeros((data["num_periods"], size))
    for j in range(data["num_processes"]):
        for t in range(data["num_periods"]):
            workforce_matrix[t, j * data["num_periods"] + t] = 1.0
    constraint = LinearConstraint(
        workforce_matrix, -np.inf, data["workforce_available"]
    )
    bounds = Bounds(np.zeros(size), data["max_staffing"].ravel())
    return bounds, constraint


def _solve_scipy_policy(data, workloads, name, max_iterations=1000):
    """Shared SLSQP implementation for nominal and expected objectives."""
    start = time.perf_counter()
    x0 = project_staffing(
        np.mean(workloads, axis=0) / data["productivity"][:, None], data
    )
    history = []

    def objective(flat_x):
        x = flat_x.reshape(data["num_processes"], data["num_periods"])
        return float(np.mean(scenario_losses(x, workloads, data)))

    def gradient(flat_x):
        x = flat_x.reshape(data["num_processes"], data["num_periods"])
        return np.mean(scenario_gradients(x, workloads, data), axis=0).ravel()

    def callback(flat_x):
        history.append(objective(flat_x))

    bounds, constraint = _scipy_constraints(data)
    result = minimize(
        objective,
        x0.ravel(),
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[constraint],
        callback=callback,
        options={"maxiter": max_iterations, "ftol": 1e-9, "disp": False},
    )
    x = project_staffing(
        result.x.reshape(data["num_processes"], data["num_periods"]), data
    )
    if not history:
        history.append(objective(x.ravel()))
    return {
        "name": name,
        "x": x,
        "iterations": int(result.nit),
        "runtime_seconds": float(time.perf_counter() - start),
        "status": "converged" if result.success else str(result.message),
        "objective_history": history,
    }


def solve_nominal(data, max_iterations=1000):
    """Optimize one point forecast using the common planner result format."""
    return _solve_scipy_policy(
        data, data["forecast"][None, :, :], "Nominal", max_iterations
    )


def solve_expected(data, max_iterations=1000):
    """Optimize mean loss over all training scenarios."""
    return _solve_scipy_policy(
        data, data["train_workloads"], "Expected", max_iterations
    )


def solve_robust_extragradient(
    data,
    step_size=0.035,
    max_iterations=15000,
    tolerance=2e-5,
    record_every=25,
    verbose=False,
):
    """Projected extragradient for min_x max_{p in simplex} sum p_s F_s(x).

    Predictor: z_half = P(z - eta G(z)).
    Corrector: z_next = P(z - eta G(z_half)).
    The corrector starts from z (not z_half), which is the defining distinction
    between extragradient and two ordinary projected-gradient steps.
    """
    start = time.perf_counter()
    scenarios = data["train_workloads"]
    x0 = project_staffing(
        np.mean(scenarios, axis=0) / data["productivity"][:, None], data
    )
    y = x0 / data["worker_scale"]
    p = np.full(data["num_train_scenarios"], 1.0 / data["num_train_scenarios"])

    history = {
        "iteration": [],
        "worst_case_loss": [],
        "primal_change": [],
        "dual_change": [],
        "natural_residual": [],
    }
    status = "maximum iterations reached"

    for k in range(1, max_iterations + 1):
        grad_y, operator_p = saddle_operator(y, p, data)
        y_half = project_scaled_staffing(y - step_size * grad_y, data)
        p_half = project_simplex(p - step_size * operator_p)

        grad_y_half, operator_p_half = saddle_operator(y_half, p_half, data)
        y_new = project_scaled_staffing(y - step_size * grad_y_half, data)
        p_new = project_simplex(p - step_size * operator_p_half)

        primal_change = float(np.linalg.norm(y_new - y))
        dual_change = float(np.linalg.norm(p_new - p))
        y, p = y_new, p_new

        if k == 1 or k % record_every == 0 or k == max_iterations:
            x = data["worker_scale"] * y
            worst_loss = float(np.max(scenario_losses(x, scenarios, data)))
            residual = natural_residual(y, p, data)
            history["iteration"].append(k)
            history["worst_case_loss"].append(worst_loss)
            history["primal_change"].append(primal_change)
            history["dual_change"].append(dual_change)
            history["natural_residual"].append(residual)
            if verbose:
                print(
                    f"EG {k:5d} | worst loss {worst_loss:,.2f} | "
                    f"residual {residual:.3e}"
                )
            if residual < tolerance:
                status = "converged"
                break

    return {
        "name": "Robust",
        "x": data["worker_scale"] * y,
        "p": p,
        "iterations": k,
        "runtime_seconds": float(time.perf_counter() - start),
        "status": status,
        "step_size": step_size,
        "history": history,
    }


def solve_robust_scipy_reference(data, max_iterations=1500):
    """Independently solve the robust epigraph model for verification only."""
    start = time.perf_counter()
    scenarios = data["train_workloads"]
    num_x = data["num_processes"] * data["num_periods"]
    x0 = project_staffing(
        np.mean(scenarios, axis=0) / data["productivity"][:, None], data
    )
    scale = data["objective_scale"]
    worst0 = float(np.max(scenario_losses(x0, scenarios, data)))
    # q is the epigraph value divided by objective_scale. This improves SLSQP
    # conditioning while leaving the physical optimization problem unchanged.
    z0 = np.concatenate([x0.ravel(), [worst0 / scale + 1.0 / scale]])

    def objective(z):
        return float(z[-1])

    def objective_jac(z):
        answer = np.zeros_like(z)
        answer[-1] = 1.0
        return answer

    def epigraph_constraints(z):
        x = z[:num_x].reshape(data["num_processes"], data["num_periods"])
        return z[-1] - scenario_losses(x, scenarios, data) / scale

    def epigraph_jac(z):
        x = z[:num_x].reshape(data["num_processes"], data["num_periods"])
        gradients = scenario_gradients(x, scenarios, data).reshape(len(scenarios), -1)
        return np.column_stack([-gradients / scale, np.ones(len(scenarios))])

    staffing_bounds, staffing_constraint = _scipy_constraints(data)
    workforce_with_epigraph = LinearConstraint(
        np.column_stack([staffing_constraint.A, np.zeros(data["num_periods"])]),
        staffing_constraint.lb,
        staffing_constraint.ub,
    )
    bounds = Bounds(
        np.concatenate([staffing_bounds.lb, [-np.inf]]),
        np.concatenate([staffing_bounds.ub, [np.inf]]),
    )
    result = minimize(
        objective,
        z0,
        jac=objective_jac,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            workforce_with_epigraph,
            {"type": "ineq", "fun": epigraph_constraints, "jac": epigraph_jac},
        ],
        options={"maxiter": max_iterations, "ftol": 1e-8, "disp": False},
    )
    x = project_staffing(
        result.x[:num_x].reshape(data["num_processes"], data["num_periods"]), data
    )
    return {
        "x": x,
        "objective": float(np.max(scenario_losses(x, scenarios, data))),
        "iterations": int(result.nit),
        "runtime_seconds": float(time.perf_counter() - start),
        "status": "converged" if result.success else str(result.message),
    }
