from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    baseline_path = Path("results/tables/generated/baseline_metrics.json")
    ensemble_path = Path("results/tables/generated/ensemble_metrics.json")
    output_path = Path("results/tables/generated/baseline_vs_ensemble.json")

    baseline = load_json(baseline_path)
    ensemble = load_json(ensemble_path)

    comparison = {
        "baseline": baseline,
        "ensemble": ensemble,
        "delta": {
            "accuracy": ensemble["accuracy"] - baseline["accuracy"],
            "nll": ensemble["nll"] - baseline["nll"],
            "brier": ensemble["brier"] - baseline["brier"],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("Baseline vs Ensemble Comparison")
    print("-" * 40)
    print(f"Baseline accuracy: {baseline['accuracy']:.6f}")
    print(f"Ensemble accuracy: {ensemble['accuracy']:.6f}")
    print(f"Delta accuracy:    {comparison['delta']['accuracy']:+.6f}")
    print()
    print(f"Baseline NLL:      {baseline['nll']:.6f}")
    print(f"Ensemble NLL:      {ensemble['nll']:.6f}")
    print(f"Delta NLL:         {comparison['delta']['nll']:+.6f}")
    print()
    print(f"Baseline Brier:    {baseline['brier']:.6f}")
    print(f"Ensemble Brier:    {ensemble['brier']:.6f}")
    print(f"Delta Brier:       {comparison['delta']['brier']:+.6f}")
    print()
    print(f"Saved comparison to: {output_path}")


if __name__ == "__main__":
    main()
