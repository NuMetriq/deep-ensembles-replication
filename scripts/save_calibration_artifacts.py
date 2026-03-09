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


def save_markdown_table(rows: list[list[str]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]

    def fmt_row(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )

    header = fmt_row(rows[0])
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = "\n".join(fmt_row(row) for row in rows[1:])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(sep + "\n")
        f.write(body + "\n")

    return output_path


def save_csv(rows: list[list[object]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return output_path


def save_clean_comparison_tables(
    baseline: dict,
    ensemble: dict,
    csv_path: str | Path,
    md_path: str | Path,
) -> tuple[Path, Path]:
    rows_csv = [
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
        ["ece", baseline["ece"], ensemble["ece"], ensemble["ece"] - baseline["ece"]],
    ]

    rows_md = [
        ["Metric", "Baseline", "Ensemble", "Delta (Ensemble - Baseline)"],
        [
            "Accuracy",
            f"{baseline['accuracy']:.6f}",
            f"{ensemble['accuracy']:.6f}",
            f"{ensemble['accuracy'] - baseline['accuracy']:+.6f}",
        ],
        [
            "NLL",
            f"{baseline['nll']:.6f}",
            f"{ensemble['nll']:.6f}",
            f"{ensemble['nll'] - baseline['nll']:+.6f}",
        ],
        [
            "Brier",
            f"{baseline['brier']:.6f}",
            f"{ensemble['brier']:.6f}",
            f"{ensemble['brier'] - baseline['brier']:+.6f}",
        ],
        [
            "ECE",
            f"{baseline['ece']:.6f}",
            f"{ensemble['ece']:.6f}",
            f"{ensemble['ece'] - baseline['ece']:+.6f}",
        ],
    ]

    return save_csv(rows_csv, csv_path), save_markdown_table(rows_md, md_path)


def save_shift_comparison_tables(
    clean_baseline: dict,
    clean_ensemble: dict,
    shifted_baseline: dict,
    shifted_ensemble: dict,
    csv_path: str | Path,
    md_path: str | Path,
) -> tuple[Path, Path]:
    rows_csv = [
        ["model", "condition", "accuracy", "nll", "brier", "ece"],
        [
            "baseline",
            "clean",
            clean_baseline["accuracy"],
            clean_baseline["nll"],
            clean_baseline["brier"],
            clean_baseline["ece"],
        ],
        [
            "baseline",
            "shifted",
            shifted_baseline["accuracy"],
            shifted_baseline["nll"],
            shifted_baseline["brier"],
            shifted_baseline["ece"],
        ],
        [
            "ensemble",
            "clean",
            clean_ensemble["accuracy"],
            clean_ensemble["nll"],
            clean_ensemble["brier"],
            clean_ensemble["ece"],
        ],
        [
            "ensemble",
            "shifted",
            shifted_ensemble["accuracy"],
            shifted_ensemble["nll"],
            shifted_ensemble["brier"],
            shifted_ensemble["ece"],
        ],
    ]

    rows_md = [
        ["Model", "Condition", "Accuracy", "NLL", "Brier", "ECE"],
        [
            "Baseline",
            "Clean",
            f"{clean_baseline['accuracy']:.6f}",
            f"{clean_baseline['nll']:.6f}",
            f"{clean_baseline['brier']:.6f}",
            f"{clean_baseline['ece']:.6f}",
        ],
        [
            "Baseline",
            "Shifted",
            f"{shifted_baseline['accuracy']:.6f}",
            f"{shifted_baseline['nll']:.6f}",
            f"{shifted_baseline['brier']:.6f}",
            f"{shifted_baseline['ece']:.6f}",
        ],
        [
            "Ensemble",
            "Clean",
            f"{clean_ensemble['accuracy']:.6f}",
            f"{clean_ensemble['nll']:.6f}",
            f"{clean_ensemble['brier']:.6f}",
            f"{clean_ensemble['ece']:.6f}",
        ],
        [
            "Ensemble",
            "Shifted",
            f"{shifted_ensemble['accuracy']:.6f}",
            f"{shifted_ensemble['nll']:.6f}",
            f"{shifted_ensemble['brier']:.6f}",
            f"{shifted_ensemble['ece']:.6f}",
        ],
    ]

    return save_csv(rows_csv, csv_path), save_markdown_table(rows_md, md_path)


def save_ece_comparison_plot(
    clean_baseline: dict,
    clean_ensemble: dict,
    shifted_baseline: dict,
    shifted_ensemble: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [
        "Baseline\nClean",
        "Baseline\nShifted",
        "Ensemble\nClean",
        "Ensemble\nShifted",
    ]
    values = [
        clean_baseline["ece"],
        shifted_baseline["ece"],
        clean_ensemble["ece"],
        shifted_ensemble["ece"],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, edgecolor="black")
    plt.ylabel("ECE")
    plt.title("Calibration Comparison Across Clean and Shifted Conditions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    clean_baseline = load_json("results/tables/generated/baseline_metrics.json")
    clean_ensemble = load_json("results/tables/generated/ensemble_metrics.json")
    shifted_baseline = load_json(
        "results/tables/generated/shifted_baseline_metrics.json"
    )
    shifted_ensemble = load_json(
        "results/tables/generated/shifted_ensemble_metrics.json"
    )

    clean_csv, clean_md = save_clean_comparison_tables(
        clean_baseline,
        clean_ensemble,
        csv_path="results/tables/generated/calibration_comparison_clean.csv",
        md_path="results/tables/generated/calibration_comparison_clean.md",
    )

    shift_csv, shift_md = save_shift_comparison_tables(
        clean_baseline,
        clean_ensemble,
        shifted_baseline,
        shifted_ensemble,
        csv_path="results/tables/generated/calibration_comparison_shift.csv",
        md_path="results/tables/generated/calibration_comparison_shift.md",
    )

    ece_plot = save_ece_comparison_plot(
        clean_baseline,
        clean_ensemble,
        shifted_baseline,
        shifted_ensemble,
        output_path="results/figures/generated/ece_clean_vs_shifted.png",
    )

    print(f"Saved clean comparison CSV to: {clean_csv}")
    print(f"Saved clean comparison markdown to: {clean_md}")
    print(f"Saved shift comparison CSV to: {shift_csv}")
    print(f"Saved shift comparison markdown to: {shift_md}")
    print(f"Saved ECE comparison plot to: {ece_plot}")


if __name__ == "__main__":
    main()
