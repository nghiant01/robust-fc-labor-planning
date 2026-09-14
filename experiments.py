"""Business metrics and in-sample / held-out stress evaluation."""

import numpy as np
import pandas as pd

from model import common_cost, scenario_losses


def evaluate_plan(x, workloads, productivity, data):
    """Evaluate one fixed staffing plan against a scenario collection."""
    workloads = np.asarray(workloads, dtype=float)
    rates = np.asarray(productivity, dtype=float)
    if rates.ndim == 1:
        rates = np.tile(rates, (len(workloads), 1))
    capacity = rates[:, :, None] * np.asarray(x)[None, :, :]
    shortage = np.maximum(workloads - capacity, 0.0)
    processed = workloads - shortage
    scenario_shortage = shortage.sum(axis=(1, 2))
    # "Single period" aggregates unmet units across all four processes.
    shortage_by_period = shortage.sum(axis=1)
    losses = scenario_losses(x, workloads, data, productivity=rates)

    supplied_capacity = np.sum(capacity)
    utilization = np.sum(processed) / supplied_capacity if supplied_capacity > 0 else 0.0
    service_level = 1.0 - np.sum(shortage) / np.sum(workloads)

    return {
        "staffing_labor_cost": float(np.sum(data["labor_cost"] * x)),
        "total_allocated_labor_hours": float(np.sum(x)),
        "adjustment_cost": float(
            common_cost(x, data) - np.sum(data["labor_cost"] * x)
        ),
        "average_shortage_units": float(np.mean(scenario_shortage)),
        "worst_case_shortage_units": float(np.max(scenario_shortage)),
        "maximum_single_period_shortage_units": float(np.max(shortage_by_period)),
        "service_level": float(service_level),
        "utilization": float(utilization),
        "average_total_loss": float(np.mean(losses)),
        "worst_case_total_loss": float(np.max(losses)),
    }


def evaluate_all(plans, data, stress_tests):
    """Build the long-form summary used by the CSV, console, and plots."""
    rows = []
    evaluation_sets = {
        "Training scenarios": {
            "workloads": data["train_workloads"],
            "productivity": data["productivity"],
            "sample": "In-sample",
        }
    }
    for regime, values in stress_tests.items():
        evaluation_sets[regime] = {
            **values,
            "sample": "Out-of-sample",
        }

    for regime, values in evaluation_sets.items():
        for planner, x in plans.items():
            row = {
                "sample": values["sample"],
                "regime": regime,
                "planner": planner,
            }
            row.update(
                evaluate_plan(
                    x, values["workloads"], values["productivity"], data
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_overall_stress_summary(plans, data, stress_tests):
    """Pool all independent regimes, retaining each scenario's true rates."""
    workloads = np.concatenate([v["workloads"] for v in stress_tests.values()], axis=0)
    productivity = np.concatenate([v["productivity"] for v in stress_tests.values()], axis=0)
    rows = []
    for planner, x in plans.items():
        row = {"planner": planner}
        row.update(evaluate_plan(x, workloads, productivity, data))
        rows.append(row)
    return pd.DataFrame(rows)

