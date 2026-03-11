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
    clean = load_json("results/tables/generated/mc_dropout_metrics.json")
    shifted = load_json("results/tables/generated/shifted_mc_dropout_metrics.json")

    print("MC Dropout Clean vs Shifted Comparison")
    print("-" * 45)
    print(f"Clean accuracy:   {clean['accuracy']:.6f}")
    print(f"Shifted accuracy: {shifted['accuracy']:.6f}")
    print(f"Δ accuracy:       {shifted['accuracy'] - clean['accuracy']:+.6f}")
    print()
    print(f"Clean NLL:        {clean['nll']:.6f}")
    print(f"Shifted NLL:      {shifted['nll']:.6f}")
    print(f"Δ NLL:            {shifted['nll'] - clean['nll']:+.6f}")
    print()
    print(f"Clean Brier:      {clean['brier']:.6f}")
    print(f"Shifted Brier:    {shifted['brier']:.6f}")
    print(f"Δ Brier:          {shifted['brier'] - clean['brier']:+.6f}")
    print()
    print(f"Clean ECE:        {clean['ece']:.6f}")
    print(f"Shifted ECE:      {shifted['ece']:.6f}")
    print(f"Δ ECE:            {shifted['ece'] - clean['ece']:+.6f}")


if __name__ == "__main__":
    main()
