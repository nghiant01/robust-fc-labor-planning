"""A small set of professional Matplotlib figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_forecast(data, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    hours = np.arange(1, data["num_periods"] + 1)
    for name, workload in zip(data["process_names"], data["forecast"]):
        ax.plot(hours, workload, marker="o", linewidth=2, label=name)
    ax.set(xlabel="Planning hour", ylabel="Forecast workload (units)",
           title="Synthetic intraday workload forecast")
    ax.set_xticks(hours)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    _save(fig, path)


def plot_staffing(plans, data, path):
    """Compare planners process by process so small reallocations stay visible."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    hours = np.arange(1, data["num_periods"] + 1)
    for j, (ax, process) in enumerate(zip(axes.ravel(), data["process_names"])):
        for planner, x in plans.items():
            ax.plot(hours, x[j], marker="o", linewidth=1.8, label=planner)
        ax.axhline(
            data["max_staffing"][j, 0], linestyle="--", color="black",
            linewidth=1.1, label="Process maximum"
        )
        ax.set_title(process)
        ax.set_ylabel("Workers")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Planning hour")
        ax.set_xticks(hours)
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.suptitle("Intraday staffing allocation by planner", y=1.01, fontsize=14)
    _save(fig, path)


def plot_stress_service(summary, path):
    stress = summary[summary["sample"] == "Out-of-sample"].copy()
    regimes = list(dict.fromkeys(stress["regime"]))
    planners = list(dict.fromkeys(stress["planner"]))
    positions = np.arange(len(regimes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, planner in enumerate(planners):
        values = [
            100.0 * stress.loc[
                (stress["regime"] == regime) & (stress["planner"] == planner),
                "service_level",
            ].iloc[0]
            for regime in regimes
        ]
        ax.bar(positions + (i - 1) * width, values, width, label=planner)
    ax.set_xticks(positions)
    ax.set_xticklabels(regimes, rotation=18, ha="right")
    ax.set_ylabel("Workload-weighted service level (%)")
    ax.set_title("Independent stress-test performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, path)


def plot_robust_convergence(result, path):
    history = result["history"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["iteration"], history["worst_case_loss"], linewidth=1.8)
    axes[0].set(xlabel="Iteration", ylabel="Worst training loss ($)",
                title="Robust objective")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(
        history["iteration"], history["natural_residual"], linewidth=1.8
    )
    axes[1].set(xlabel="Iteration", ylabel="Natural residual",
                title="Saddle-point residual")
    axes[1].grid(alpha=0.25)
    _save(fig, path)
