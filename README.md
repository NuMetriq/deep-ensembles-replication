# Deep Ensembles Replication



A PyTorch replication of *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (Lakshminarayanan, Pritzel, Blundell, 2017), with a focus on classification performance, predictive uncertainty, and calibration.



\## Overview



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

├── configs/

│   └── mnist.yaml

├── src/

│   ├── data.py

│   ├── model.py

│   ├── train.py

│   ├── evaluate.py

│   ├── ensemble.py

│   └── metrics.py

├── scripts/

│   ├── train_single.py

│   ├── train_ensemble.py

│   └── evaluate_mnist.py

├── results/

│   ├── figures/

│   └── tables/

└── report/

   └── replication_report.md

```



## Current Status



This project is currently in the setup and baseline implementation phase.



### Milestone: v0.1.0 - Baseline Replication Setup

* ✅ Create GitHub repository
* ✅ Create local project structure
* ✅ Add `.gitignore`
* ✅ Add initial README
* ❌ Set up environment and dependencies
* ❌ Implement MNIST data pipeline
* ❌ Implement baseline classifier
* ❌ Implement evaluation metrics
* ❌ Implement reliability diagram
* ❌ Add runnable training and evaluation scripts



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



## Usage



Instructions will be added as components are implemented.



Planned commands:



### Train baseline model

```bash

python scripts/train_single.py

```



### Train ensemble

```bash

python scripts/train_ensemble.py

```



### Evaluate model(s)

```bash

python scripts/evaluate_mnist.py

```



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

