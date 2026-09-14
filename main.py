"""Run the complete reproducible labor-planning experiment."""

from pathlib import Path

import numpy as np
import pandas as pd

from data import generate_data, generate_stress_tests
from experiments import build_overall_stress_summary, evaluate_all
from model import (
    check_staffing_feasibility,
    finite_difference_gradient_check,
    project_capped_simplex,
    project_simplex,
    scenario_losses,
)
from plotting import (
    plot_forecast,
    plot_robust_convergence,
    plot_staffing,
    plot_stress_service,
)
from solvers import (
    solve_expected,
    solve_nominal,
    solve_robust_extragradient,
    solve_robust_scipy_reference,
)


SEED = 2026
ROBUST_SETTINGS = {
    "step_size": 0.035,
    "max_iterations": 15000,
    "tolerance": 2e-5,
    "record_every": 25,
    "verbose": False,
}


def run_numerical_checks(data):
    gradient = finite_difference_gradient_check(data)
    if not gradient["passed"]:
        raise RuntimeError(f"Finite-difference gradient check failed: {gradient}")

    simplex = project_simplex(np.array([0.8, -0.3, 1.7, 0.1]))
    assert np.all(simplex >= -1e-12) and abs(simplex.sum() - 1.0) < 1e-12
    assert np.allclose(simplex, np.array([0.05, 0.0, 0.95, 0.0]), atol=1e-12)

    capped = project_capped_simplex(
        np.array([9.0, -1.0, 5.0, 7.0]), np.array([6.0, 6.0, 6.0, 6.0]), 12.0
    )
    assert np.all(capped >= -1e-12) and np.all(capped <= 6.0 + 1e-12)
    assert capped.sum() <= 12.0 + 1e-10
    assert np.allclose(capped, np.array([6.0, 0.0, 2.0, 4.0]), atol=1e-10)
    print(f"Gradient and projection checks passed: {gradient}")
    return gradient


def main():
    root = Path(__file__).resolve().parent
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    data = generate_data(seed=SEED)
    stress_tests = generate_stress_tests(data, scenarios_per_regime=200)
    gradient = run_numerical_checks(data)

    nominal = solve_nominal(data)
    expected = solve_expected(data)
    robust = solve_robust_extragradient(data, **ROBUST_SETTINGS)
    for result in (nominal, expected, robust):
        if result["status"] != "converged":
            raise RuntimeError(f"{result['name']} solver did not converge: {result['status']}")
    plans = {"Nominal": nominal["x"], "Expected": expected["x"], "Robust": robust["x"]}

    for planner, x in plans.items():
        diagnostics = check_staffing_feasibility(x, data)
        if not diagnostics["feasible"] or not np.all(np.isfinite(x)):
            raise RuntimeError(f"{planner} plan failed feasibility checks: {diagnostics}")

    if np.any(robust["p"] < -1e-10) or abs(robust["p"].sum() - 1.0) > 1e-10:
        raise RuntimeError("Robust dual vector is not on the probability simplex.")

    summary = evaluate_all(plans, data, stress_tests)
    overall = build_overall_stress_summary(plans, data, stress_tests)
    if not np.all(np.isfinite(summary.select_dtypes(include=[np.number]))):
        raise RuntimeError("Non-finite value found in result metrics.")
    if not summary["service_level"].between(0.0, 1.0).all():
        raise RuntimeError("Service level outside [0,1].")
    if not summary["utilization"].between(0.0, 1.0 + 1e-12).all():
        raise RuntimeError("Utilization outside [0,1].")

    summary.to_csv(results_dir / "summary.csv", index=False)
    overall.to_csv(results_dir / "overall_stress_summary.csv", index=False)

    reference = solve_robust_scipy_reference(data)
    if reference["status"] != "converged":
        raise RuntimeError(f"SciPy robust reference failed: {reference['status']}")
    robust_objective = float(
        np.max(scenario_losses(robust["x"], data["train_workloads"], data))
    )
    validation = {
        "gradient_check": str(gradient),
        "nominal_status": nominal["status"],
        "expected_status": expected["status"],
        "robust_status": robust["status"],
        "robust_iterations": robust["iterations"],
        "robust_final_residual": robust["history"]["natural_residual"][-1],
        "extragradient_objective": robust_objective,
        "scipy_reference_status": reference["status"],
        "scipy_reference_objective": reference["objective"],
        "relative_objective_gap": (
            (robust_objective - reference["objective"])
            / max(1.0, abs(reference["objective"]))
        ),
    }
    # A tiny one-row CSV keeps the verification evidence machine-readable.
    pd.DataFrame([validation]).to_csv(results_dir / "solver_validation.csv", index=False)

    plot_forecast(data, figures_dir / "workload_forecast.png")
    plot_staffing(plans, data, figures_dir / "staffing_allocation.png")
    plot_stress_service(summary, figures_dir / "stress_test_service_level.png")
    plot_robust_convergence(robust, figures_dir / "robust_convergence.png")

    display_columns = [
        "planner", "staffing_labor_cost", "total_allocated_labor_hours",
        "average_shortage_units", "worst_case_shortage_units", "service_level",
        "utilization", "average_total_loss", "worst_case_total_loss",
    ]
    print("\nPooled independent stress-test results:")
    print(overall[display_columns].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\nSolver validation:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    print("\nCreated results/summary.csv and four figures.")


if __name__ == "__main__":
    main()
