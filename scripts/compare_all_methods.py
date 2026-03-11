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


def save_csv(rows: list[list[object]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return output_path


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


def save_clean_comparison(
    baseline: dict,
    ensemble: dict,
    mc_dropout: dict,
    csv_path: str | Path,
    md_path: str | Path,
) -> tuple[Path, Path]:
    rows_csv = [
        ["method", "accuracy", "nll", "brier", "ece"],
        [
            "baseline",
            baseline["accuracy"],
            baseline["nll"],
            baseline["brier"],
            baseline["ece"],
        ],
        [
            "ensemble",
            ensemble["accuracy"],
            ensemble["nll"],
            ensemble["brier"],
            ensemble["ece"],
        ],
        [
            "mc_dropout",
            mc_dropout["accuracy"],
            mc_dropout["nll"],
            mc_dropout["brier"],
            mc_dropout["ece"],
        ],
    ]

    rows_md = [
        ["Method", "Accuracy", "NLL", "Brier", "ECE"],
        [
            "Baseline",
            f"{baseline['accuracy']:.6f}",
            f"{baseline['nll']:.6f}",
            f"{baseline['brier']:.6f}",
            f"{baseline['ece']:.6f}",
        ],
        [
            "Ensemble",
            f"{ensemble['accuracy']:.6f}",
            f"{ensemble['nll']:.6f}",
            f"{ensemble['brier']:.6f}",
            f"{ensemble['ece']:.6f}",
        ],
        [
            "MC Dropout",
            f"{mc_dropout['accuracy']:.6f}",
            f"{mc_dropout['nll']:.6f}",
            f"{mc_dropout['brier']:.6f}",
            f"{mc_dropout['ece']:.6f}",
        ],
    ]

    return save_csv(rows_csv, csv_path), save_markdown_table(rows_md, md_path)


def save_shifted_comparison(
    shifted_baseline: dict,
    shifted_ensemble: dict,
    shifted_mc_dropout: dict,
    csv_path: str | Path,
    md_path: str | Path,
) -> tuple[Path, Path]:
    rows_csv = [
        ["method", "accuracy", "nll", "brier", "ece"],
        [
            "baseline_shifted",
            shifted_baseline["accuracy"],
            shifted_baseline["nll"],
            shifted_baseline["brier"],
            shifted_baseline["ece"],
        ],
        [
            "ensemble_shifted",
            shifted_ensemble["accuracy"],
            shifted_ensemble["nll"],
            shifted_ensemble["brier"],
            shifted_ensemble["ece"],
        ],
        [
            "mc_dropout_shifted",
            shifted_mc_dropout["accuracy"],
            shifted_mc_dropout["nll"],
            shifted_mc_dropout["brier"],
            shifted_mc_dropout["ece"],
        ],
    ]

    rows_md = [
        ["Method", "Accuracy", "NLL", "Brier", "ECE"],
        [
            "Baseline (Shifted)",
            f"{shifted_baseline['accuracy']:.6f}",
            f"{shifted_baseline['nll']:.6f}",
            f"{shifted_baseline['brier']:.6f}",
            f"{shifted_baseline['ece']:.6f}",
        ],
        [
            "Ensemble (Shifted)",
            f"{shifted_ensemble['accuracy']:.6f}",
            f"{shifted_ensemble['nll']:.6f}",
            f"{shifted_ensemble['brier']:.6f}",
            f"{shifted_ensemble['ece']:.6f}",
        ],
        [
            "MC Dropout (Shifted)",
            f"{shifted_mc_dropout['accuracy']:.6f}",
            f"{shifted_mc_dropout['nll']:.6f}",
            f"{shifted_mc_dropout['brier']:.6f}",
            f"{shifted_mc_dropout['ece']:.6f}",
        ],
    ]

    return save_csv(rows_csv, csv_path), save_markdown_table(rows_md, md_path)


def save_metric_plot(
    values: list[float],
    labels: list[str],
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, edgecolor="black")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    baseline = load_json("results/tables/generated/baseline_metrics.json")
    ensemble = load_json("results/tables/generated/ensemble_metrics.json")
    mc_dropout = load_json("results/tables/generated/mc_dropout_metrics.json")

    shifted_baseline = load_json(
        "results/tables/generated/shifted_baseline_metrics.json"
    )
    shifted_ensemble = load_json(
        "results/tables/generated/shifted_ensemble_metrics.json"
    )
    shifted_mc_dropout = load_json(
        "results/tables/generated/shifted_mc_dropout_metrics.json"
    )

    clean_csv, clean_md = save_clean_comparison(
        baseline,
        ensemble,
        mc_dropout,
        csv_path="results/tables/generated/three_way_comparison_clean.csv",
        md_path="results/tables/generated/three_way_comparison_clean.md",
    )

    shifted_csv, shifted_md = save_shifted_comparison(
        shifted_baseline,
        shifted_ensemble,
        shifted_mc_dropout,
        csv_path="results/tables/generated/three_way_comparison_shifted.csv",
        md_path="results/tables/generated/three_way_comparison_shifted.md",
    )

    clean_ece_plot = save_metric_plot(
        values=[baseline["ece"], ensemble["ece"], mc_dropout["ece"]],
        labels=["Baseline", "Ensemble", "MC Dropout"],
        title="Clean MNIST ECE Comparison",
        ylabel="ECE",
        output_path="results/figures/generated/three_way_ece_clean.png",
    )

    shifted_ece_plot = save_metric_plot(
        values=[
            shifted_baseline["ece"],
            shifted_ensemble["ece"],
            shifted_mc_dropout["ece"],
        ],
        labels=["Baseline", "Ensemble", "MC Dropout"],
        title="Shifted MNIST ECE Comparison",
        ylabel="ECE",
        output_path="results/figures/generated/three_way_ece_shifted.png",
    )

    clean_nll_plot = save_metric_plot(
        values=[baseline["nll"], ensemble["nll"], mc_dropout["nll"]],
        labels=["Baseline", "Ensemble", "MC Dropout"],
        title="Clean MNIST NLL Comparison",
        ylabel="NLL",
        output_path="results/figures/generated/three_way_nll_clean.png",
    )

    shifted_nll_plot = save_metric_plot(
        values=[
            shifted_baseline["nll"],
            shifted_ensemble["nll"],
            shifted_mc_dropout["nll"],
        ],
        labels=["Baseline", "Ensemble", "MC Dropout"],
        title="Shifted MNIST NLL Comparison",
        ylabel="NLL",
        output_path="results/figures/generated/three_way_nll_shifted.png",
    )

    print(f"Saved clean comparison CSV to: {clean_csv}")
    print(f"Saved clean comparison markdown to: {clean_md}")
    print(f"Saved shifted comparison CSV to: {shifted_csv}")
    print(f"Saved shifted comparison markdown to: {shifted_md}")
    print(f"Saved clean ECE plot to: {clean_ece_plot}")
    print(f"Saved shifted ECE plot to: {shifted_ece_plot}")
    print(f"Saved clean NLL plot to: {clean_nll_plot}")
    print(f"Saved shifted NLL plot to: {shifted_nll_plot}")


if __name__ == "__main__":
    main()
