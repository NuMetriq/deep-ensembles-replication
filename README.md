# Deep Ensembles Replication



A PyTorch replication of *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (Lakshminarayanan, Pritzel, Blundell, 2017), with a focus on classification performance, predictive uncertainty, and calibration.



## Overview



This project aims to replicate key ideas from the Deep Ensembles paper in a clean, portfolio-friendly way. The initial focus is on a manageable classification setting using MNIST, comparing a single neural network against a deep ensemble of independently trained networks.



The broader goal is to understand how deep ensembles improve uncertainty estimation and calibration while remaining simple to implement and scale.



## Paper



**Reference**  

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.  
*Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.*  
NeurIPS 2017.



## Replication Goal



The first replication target is:



> Does a 5-member deep ensemble improve classification performance and calibration on MNIST relative to a single neural network?



This project is designed as a step-by-step replication rather than an attempt to reproduce the full paper in one pass.



## Planned Scope



### Initial scope

- Train a single baseline classifier on MNIST
- Train an ensemble of 5 independently initialized classifiers
- Compare single-model vs ensemble performance
- Evaluate uncertainty-aware metrics
- Generate calibration plots



### Metrics

- Accuracy
- Negative Log-Likelihood (NLL)
- Brier Score
- Reliability Diagram
- Expected Calibration Error (optional extension)



### Possible later extensions

- MC Dropout comparison
- Adversarial training
- Out-of-distribution evaluation
- Additional datasets beyond MNIST



## Repository Structure



```text

deep-ensembles-replication/

├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── configs/
│   └── mnist.yaml
├── src/
│   ├── __init__.py
│   ├── plotting.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── ensemble.py
│   └── metrics.py
├── scripts/
│   ├── __init__.py
│   ├── compare_results.py
│   ├── evaluate_ensemble.py
│   ├── save_comparison_artifacts.py
│   ├── train_single.py
│   ├── train_ensemble.py
│   └── evaluate_mnist.py
├── tests/
│   ├── test_data.py
│   ├── test_ensemble.py
│   ├── test_evaluate.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_plotting.py
│   └── test_train.py
├── results/
│   ├── figures/
│   └── tables/
└── report/
   └── replication_report.md

```



## Environment Setup



### 1. Clone the repository



```bash

git clone https://github.com/NuMetriq/deep-ensembles-replication.git

cd deep-ensembles replication

```



### 2. Create and activate a virtual environment



#### Windows (PowerShell)

```bash

python -m venv .venv

.venv\Scripts\Activate.ps1

```



#### macOS / Linux

```bash

python -m venv .venv

source .venv/bin/activate

```



### 3. Install dependencies



```bash

pip install -r requirements.txt

```



## How to Reproduce the Results



### Train the baseline model



```bash
python -m scripts.train_single
```


### Evaluate the baseline model


```bash
python -m scripts.evaluate_mnist
```


### Train the deep ensemble


```bash
python -m scripts/train_ensemble
```


### Evaluate the deep ensemble


```bash
python -m scripts.evaluate_ensemble
```


### Generate comparison artifacts


```bash
python -m scripts.compare_results
python -m scripts.save_comparison_artifacts
```


## Results



### Experiment
For the `v0.2.0` milestone, I trained:

- **Baseline:** 1 MNIST classifier
- **Ensemble:** 5 independently initialized MNIST classifiers, with predictions combined by averaging class probabilities

### Evaluation Metrics
The main evaluation metrics were:

- Accuracy
- Negative Log-Likelihood (NLL)
- Brier Score
- Reliability Diagram

### Summary Table

| Metric | Baseline | Ensemble | Delta (Ensemble - Baseline) |
|---|---:|---:|---:|
| Accuracy | 0.990200 | 0.992400 | +0.002200 |
| NLL | 0.033565 | 0.020762 | -0.012803 |
| Brier | 0.016628 | 0.010863 | -0.005765 |

### Calibration Plots

#### Baseline Reliability Diagram
![Baseline Reliability Diagram](results/figures/generated/baseline_reliability_diagram.png)

#### Ensemble Reliability Diagram
![Ensemble Reliability Diagram](results/figures/generated/ensemble_reliability_diagram.png)

### Comparison Figure
![Baseline vs Ensemble Metrics](results/figures/generated/baseline_vs_ensemble_metrics.png)

### Interpretation

The deep ensemble improved predictive performance relative to the single-model baseline, especially on uncertainty-aware metrics.

Key observations:
- Higher accuracy indicates slightly stronger classification performance
- Lower NLL suggests better probabilistic predictions
- Lower Brier score suggests better calibrated class probabilities
- The reliability diagram provides a visual check on calibration quality

Overall, the ensemble results were directionally consistent with the central claim of the paper: independently trained deep networks, combined as an ensemble, can provide strong uncertainty estimates while remaining simple to implement.



## Key Outputs

Running the full pipeline produces:

### Tables
- `results/tables/generated/baseline_metrics.json`
- `results/tables/generated/ensemble_metrics.json`
- `results/tables/generated/baseline_vs_ensemble.json`
- `results/tables/generated/baseline_vs_ensemble.csv`

### Figures
- `results/figures/generated/baseline_reliability_diagram.png`
- `results/figures/generated/ensemble_reliability_diagram.png`
- `results/figures/generated/baseline_vs_ensemble_metrics.png`

### Checkpoints
- `checkpoints/mnist_baseline.pt`
- `checkpoints/ensemble/member_0.pt`
- `checkpoints/ensemble/member_1.pt`
- `checkpoints/ensemble/member_2.pt`
- `checkpoints/ensemble/member_3.pt`
- `checkpoints/ensemble/member_4.pt`



## Current Status

### Completed Milestones

#### Milestone: v0.1.0 - Baseline Replication Setup

- ✅ Create GitHub repository
- ✅ Create local project structure
- ✅ Add `.gitignore`
- ✅ Add initial README
- ✅ Set up environment and dependencies
- ✅ Implement MNIST data pipeline
- ✅ Implement baseline classifier
- ✅ Implement evaluation metrics
- ✅ Implement reliability diagram
- ✅ Add runnable training and evaluation scripts


#### Milestone: v0.2.0 - Deep Ensemble Replication

- ✅ Implement ensemble checkpoint saving/loading
- ✅ Implement ensemble prediction via probability averaging
- ✅ Train 5 independent ensemble members
- ✅ Evaluate ensemble performance
- ✅ Compare baseline vs ensemble results
- ✅ Save comparison artifacts
- ✅ Document results in the README


### Next Milestones


#### v0.3.0 - Extended Uncertainty Analysis

Possible next steps:
- Excpected Calibration Error (ECE)
- MC Dropout comparison
- adversarial training
- out-of-distribution evaluation
- additional datasets beyond MNIST



## Replication Philosophy



This repository is intended to be:

- faithful to the central ideas of the paper
- incremental and well-documented
- reproducible
- suitable for a GitHub portfolio



The emphasis is on understanding and communicating the method clearly, not just matching every reported number exactly.



## Roadmap



### v0.1.0 - Baseline Replication Setup



Establish the project foundation and train/evaluate a single MNIST classifier.



### v0.2.0 - Deep Ensemble Replication



Train a 5-model ensemble and compare it against the baseline on core metrics.



### v0.3.0 - Calibration and Uncertainty Analysis



Expand evaluation with stronger calibration analysis and visualizations.



### v0.4.0 - Extended Replication



Add comparisons such as MC Dropout, adversarial training, or out-of-distribution tests.



## Notes



This is an education replication project intended to deepen understanding of uncertainty estimation in deep learning and to demonstrate practical machine learning engineering on GitHub.



## License



To be added.



---

