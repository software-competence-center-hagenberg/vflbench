# VFLBench: A Practical Benchmark for Vertical Federated Learning

---
VFLBench is a practical benchmarking framework for vertical federated learning.


## Datasets

---

1. Hydraulic System I (HySys I)

    This dataset was obtained experimentally using a hydraulic test rig \cite{helwig2015}, 
    which consists of two circuits interconnected via an oil tank. The system operates by
    cyclically repeating constant load cycles, during which various process values are
    measured. The aim is to develop a regression model for predicting valve conditions.
    To replicate a vertical setup, the features are divided into two blocks based on the
    rig's configuration \cite{helwig2015}, with each block assigned to a different data holder.
    Folllowing data split, feature reduction was applied separately to each data holder's
    private data to reduce the number of features.


2. Hydraulic System II (HySys II)

   Derived from the same source as HySyS I, this dataset has the same feature split.
   However, the task is different: to predict the stable flag, which is binary.


3. Steel Fatigue Strength (SFS)
    
    This dataset includes various experimental conditions during steel preparation, 
    such as chemical composition, upstream processing details, and heat treatment \cite{agrawal2014exploration}.
    The target variable is fatigue strength. Features are vertically divided into two blocks and allocated to
    two hypothetical data holders: the first block contains chemical composition and upstream details,
    while the second block includes heat treatment information.


4. Simulated Multistage Process (SMP)

    This synthetic dataset is generated using a multistage process simulator \cite{nguyen2024p3ls}. It emulates a
    three-stage process, assuming that three distinct manufacturing companies control and possess data from each specific stage.
    The primary goal of the data federation is to construct a predictive model for the output quality of the final stage.

## Vertical Federated Learning algorithms

---

1. Privacy-preserving Partial Least Squares (P3LS)

    A federated version of Partial Least Squares (PLS), which is a technique commonly used for monitoring and controlling
    manufacturing processes. P3LS involves a PLS algorithm based on singular value decomposition (SVD) and incorporates removable,
    randomly generated masks provided by a trusted authority to protect each data holder's private information.


2. Privacy-preserving Symbolic Regression (PPSR)

    A privacy-preserving variant of Symbolic Regression (SR). PPSR employs Secure Multiparty Computation to allow parties to
    collaboratively build SR models in a vertical scenario without disclosing private data.


3. Secureboost

    A federated learning algorithm that extends the gradient boosting framework, specifically XGBoost, to enable
    collaborative model training across multiple parties without sharing raw data.  It ensures data confidentiality by 
    performing secure aggregation of local computations utilizing Homomorphic Encryption, thereby preventing sensitive
    information leakage.


4. Split Neural Network (SplitNN)

    Split learning involves dividing the network structure so that each party retains only a portion of it. These smaller
    structures combine to form a complete network model. During training, parties perform forward or backward calculations
    on their local structures and transfer the results to the next party. This way, it allows multiple data holders to
    contribute to the training of a joint model until it converges. During this process, Differential Privacy technology
    might be employed to enhance privacy protection.

## VFLBench in action

---

### Testing Environment

- Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz (64 cores)
- 384 GB System Memory

### Prerequisites

- Install Docker
- Create `vflbench` image using the Dockerfile: `docker build . -t vflbench`

### Run the paper experiments

- Open the command line
- Move to `/paper_experiments`
- Execute `bash run_all.sh` to test all methods, or execute the individual shell script (e.g., `bash run_splitnn.sh`) to
  test each method separately

## Contribution guidance

---

See the [contributing](docs/CONTRIBUTING.md) document.


## References

