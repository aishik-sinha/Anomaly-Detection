# Network Intrusion Detection with Machine Learning

A machine learning pipeline that analyzes network connection data and flags suspicious connections that may represent cyberattacks - built as a portfolio project combining data science and cybersecurity.

## What this project does

This tool takes network connection records (things like connection duration, protocol type, bytes transferred, and failed login attempts) and classifies each one as either **normal** traffic or a likely **attack**. It's a decision-support tool: it flags suspicious connections for a security analyst to review, rather than automatically blocking traffic itself.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), a widely-used benchmark dataset for intrusion detection research. It's derived from network traffic captured during a 1998 DARPA intrusion detection evaluation, and includes 41 features per connection record across both normal traffic and ~22 specific attack types (grouped into 4 broader categories: DoS, Probe, R2L, and U2R).

**Known limitation:** this dataset is from 1998. Attack patterns in it are more distinct and "obvious" than what a modern, real-world intrusion detection system would face. I address this limitation directly in the findings below.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1 | `01_load_and_explore.py` | Load the raw data, inspect structure, create a simplified binary (normal/attack) label |
| 2 | `02_preprocess.py` | Encode categorical features, scale numeric features, split into train/test |
| 3 | `03_train_baseline_model.py` | Train a Decision Tree baseline, evaluate with precision/recall/F1 |
| 4 | `04_compare_models.py` | Compare Decision Tree, Random Forest, and Isolation Forest; rank features by importance |
| 5 | `05_scoring_tool.py` | Command-line tool: train once, then score any new CSV of connection data |
| 7 | `07_test_generalization.py` | Test on NSL-KDD's separate held-out file to check real-world generalization |

## Key finding: the accuracy trap, and how I diagnosed it

My first model scored **99.8% accuracy** on a standard random 80/20 train/test split. That number is misleading, and figuring out why was the most important part of this project.

**The problem:** an 80/20 split still pulls both sets from the *same file*, so it mainly tests whether the model memorized individual rows - not whether it generalizes to genuinely new data. NSL-KDD ships with a second, independently-collected file (`KDDTest+.txt`) specifically for testing this, including several attack sub-types that never appear in the training file at all.

**Testing on that real held-out file, accuracy dropped to 76.9% and recall fell to 61.5%** - meaning the model was missing nearly 40% of real attacks on genuinely new data.

**Diagnosis:** breaking results down by specific attack type revealed the model had essentially only learned to recognize the handful of *high-volume* attack types in training (`neptune`: 41,214 examples) while almost completely failing on rare ones (`guess_passwd`: 53 examples, 0% catch rate) - a classic class imbalance problem, hidden by the fact that all attack sub-types get collapsed into one binary "attack" label.

**First fix attempt - `class_weight='balanced'` - failed.** This only balances the binary normal/attack label, which was already roughly 54/46. It had no visibility into the real imbalance, which existed one level deeper, between individual attack sub-types.

**Working fix - `sample_weight` computed from the original detailed attack labels.** By weighting each training row based on the rarity of its *specific* attack sub-type (not the collapsed binary label), the model was forced to pay real attention to rare-but-present attack types. This improved recall on the held-out set from **61.5% to 74.9%**, and moved `guess_passwd`'s catch rate from 0% to 28.9%.

| Approach | Accuracy | Recall | F1 |
|---|---|---|---|
| No balancing | 0.769 | 0.615 | 0.752 |
| `class_weight='balanced'` (binary label) | 0.762 | 0.603 | 0.743 |
| `sample_weight` (detailed attack labels) | **0.828** | **0.749** | **0.832** |

**Remaining limitation:** attack types absent from training entirely (17 of them appear only in the test file) still can't be reliably caught. No amount of reweighting can teach a model to recognize a pattern it has zero examples of - this is a fundamental limit of supervised learning, not a bug in this implementation.

## Model comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 0.997 | 0.997 | 0.997 | 0.997 |
| Random Forest | 0.998 | 0.999 | 0.997 | 0.998 |
| Random Forest (top 10 features only) | 0.998 | 0.999 | 0.996 | 0.998 |
| Isolation Forest (unsupervised) | 0.648 | 0.622 | 0.621 | 0.621 |

*(All same-file split results above; see the generalization findings for held-out performance.)*

Isolation Forest performs notably worse because it's unsupervised - it never sees labels during training and instead flags statistical outliers. This is a real tradeoff: lower accuracy, but it's the only approach here that could plausibly catch a brand-new attack type with zero labeled examples, since it doesn't rely on having seen something like it before.

## Feature importance

Using the Random Forest's built-in feature importance ranking, the strongest predictors were `src_bytes`, `dst_bytes`, `same_srv_rate`, `dst_host_srv_count`, and `serror_rate`/`srv_serror_rate` - broadly, data volume and connection-error patterns. A model using only the top 10 of 40 features achieved nearly identical performance to the full feature set, suggesting most connection-level signal is concentrated in a small subset of features.

## Tech stack

Python, pandas, scikit-learn (Decision Tree, Random Forest, Isolation Forest, preprocessing, evaluation metrics)

## How to run this

1. Download `KDDTrain+.txt` and `KDDTest+.txt` from [Kaggle](https://www.kaggle.com/datasets/harivmv/nsl-kdd-dataset) or the [NSL-KDD GitHub mirror](https://github.com/jmnwong/NSL-KDD-Dataset), place them in this folder
2. `pip install pandas scikit-learn`
3. Run the scripts in order: `python 01_load_and_explore.py`, then `02_preprocess.py`, `03_train_baseline_model.py`, `04_compare_models.py`
4. To score new data: `python 05_scoring_tool.py <path_to_csv>`
5. To test generalization on the held-out set: `python 07_test_generalization.py`

## What I'd improve with more time

- Extend `sample_weight` to the scoring tool's Isolation Forest path, and re-test whether unsupervised detection helps specifically on the attack types the supervised model still misses
- Try multi-class prediction (attack category or specific type, not just binary) - this would let importance-based class balancing work at the level where the imbalance actually lives
- Validate against a more modern dataset (e.g. CICIDS2017) to check whether these findings hold up outside 1998-era attack patterns
- Add a lightweight web interface (Streamlit) for a more visual demo
