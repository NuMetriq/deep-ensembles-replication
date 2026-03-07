from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_comparison_csv(
    baseline: dict,
    ensemble: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        ["metric", "baseline", "ensemble", "delta_ensemble_minus_baseline"],
        [
            "accuracy",
            baseline["accuracy"],
            ensemble["accuracy"],
            ensemble["accuracy"] - baseline["accuracy"],
        ],
        ["nll", baseline["nll"], ensemble["nll"], ensemble["nll"] - baseline["nll"]],
        [
            "brier",
            baseline["brier"],
            ensemble["brier"],
            ensemble["brier"] - baseline["brier"],
        ],
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return output_path


def save_metrics_comparison_plot(
    baseline: dict,
    ensemble: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["accuracy", "nll", "brier"]
    baseline_vals = [baseline[m] for m in metrics]
    ensemble_vals = [ensemble[m] for m in metrics]

    x = range(len(metrics))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([i - width / 2 for i in x], baseline_vals, width=width, label="Baseline")
    plt.bar([i + width / 2 for i in x], ensemble_vals, width=width, label="Ensemble")

    plt.xticks(list(x), metrics)
    plt.ylabel("Value")
    plt.title("Baseline vs Ensemble Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def save_markdown_summary(
    baseline: dict,
    ensemble: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "| Metric | Baseline | Ensemble | Delta (Ensemble - Baseline) |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {baseline['accuracy']:.6f} | {ensemble['accuracy']:.6f} | {ensemble['accuracy'] - baseline['accuracy']:+.6f} |",
        f"| NLL | {baseline['nll']:.6f} | {ensemble['nll']:.6f} | {ensemble['nll'] - baseline['nll']:+.6f} |",
        f"| Brier | {baseline['brier']:.6f} | {ensemble['brier']:.6f} | {ensemble['brier'] - baseline['brier']:+.6f} |",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def main() -> None:
    baseline_path = Path("results/tables/generated/baseline_metrics.json")
    ensemble_path = Path("results/tables/generated/ensemble_metrics.json")

    comparison_csv_path = Path("results/tables/generated/baseline_vs_ensemble.csv")
    comparison_plot_path = Path(
        "results/figures/generated/baseline_vs_ensemble_metrics.png"
    )

    baseline = load_json(baseline_path)
    ensemble = load_json(ensemble_path)

    markdown_path = Path("results/tables/generated/baseline_vs_ensemble.md")
    md_path = save_markdown_summary(baseline, ensemble, markdown_path)
    print(f"Saved markdown summary to: {md_path}")

    csv_path = save_comparison_csv(baseline, ensemble, comparison_csv_path)
    fig_path = save_metrics_comparison_plot(baseline, ensemble, comparison_plot_path)

    print(f"Saved comparison CSV to: {csv_path}")
    print(f"Saved comparison figure to: {fig_path}")


if __name__ == "__main__":
    main()
