# FlexTab
[![arXiv](https://img.shields.io/badge/arXiv-2506.10707-G.svg)](https://arxiv.org/abs/2606.30336) [![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/flextab)](https://api.reuse.software/info/github.com/SAP-samples/flextab)


## Description

Implementation of the deep learning models with the inference pipeline described in the paper ["FlexTab: A Flexible Encoder-Decoder Architecture for In-Context Learning Across Diverse Tabular Tasks"](https://arxiv.org/abs/2606.30336).

![logo](./architecture.png)

## Abstract

We introduce FlexTab, a flexible encoder-decoder architecture for in-context learning on tabular data that pairs a single, task-agnostic encoder with a suite of task-specific decoders. Unlike existing tabular in-context learners, which entangle feature representations with a specific prediction target, our design produces target-agnostic row embeddings that can be leveraged across a wide range of downstream tasks within a table-native in-context learning setup. We demonstrate this flexibility on six distinct problems: classification, regression, anomaly detection, clustering, entity matching, and entity classification in relational databases. Both the encoder and the task-specific decoders are trained on a large corpus of real-world, unlabeled tables. FlexTab achieves state-of-the-art performance on classification, regression, anomaly detection and entity matching, while remaining competitive with specialized models on entity classification in a relational setting. These results demonstrate that a single shared encoder, paired with task-specific decoders, can serve as an effective general-purpose backbone for diverse tabular prediction problems. Model checkpoints are publicly available on [Hugging Face](https://huggingface.co/SAP/FlexTab).

## Citations

If you use this model in your research or want to refer to our work, please cite:
```
@inproceedings{
    flextab,
    title={{FlexTab}: A Flexible Encoder-Decoder Architecture for In-Context Learning Across Diverse Tabular Tasks},
    author={Marek Polewczyk and Maximilian Schambach and Marco Spinaci and Sam Thelin and Johannes Höhne},
    booktitle={Advances in Neural Information Processing Systems},
    year={2026},
    url={https://arxiv.org/abs/2606.30336}
}
```

## Requirements

This project uses model checkpoints available on https://huggingface.co/SAP/FlexTab that are automatically downloaded when running the model.
Note that downloading the model checkpoints requires logging in to Hugging Face. See the [instructions](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) for details.

The requirements are detailed in the `pyproject.toml` file for Python 3.12 version.

Local installation in virtual environment:
```bash
uv sync
source .venv/bin/activate
```

Installation from source:
```bash
pip install git+https://github.com/SAP-samples/flextab
```

## Basic Usage

The model supports classification, regression, matching, anomaly detection, clustering, and entity classification in relational databases tasks. It accepts input data in the form of a pandas DataFrame or a NumPy array. No preprocessing is required, column names and cell values are automatically embedded using an LLM, and any missing values are handled correctly.

For best performance, use a GPU with at least 80 GB of memory and set the context size to 8192. For large tables, it is recommended to use a bagging factor of 8.

For a light-weight, faster model with lower requirements on the GPU, try lowering the context size to e.g. 2048 and set the bagging factor to 1.


### Basic test

To test the environment is correctly setup and do a basic end-to-end testing, run
```bash
uv run --extra test python test.py
```

It should return the following results:
```
CLASSIFICATION
Accuracy 98.60%
AUROC 99.92%

REGRESSION
R² Score: 49.62%

MATCHING
Accuracy 99.63%
F1 Score 98.89%
AUROC 99.96%

EMBEDDING
Self-context shape: (30, 768)
Context+query shape: (50, 768)
Context invariance max diff: 0.00e+00

ANOMALY DETECTION
Unsupervised AUROC = 86.23%
One-class AUROC = 97.73%
Semi-supervised la=1% (1/114 annotated): FlexTab AUROC = 55.68%, FlexTab-AD AUROC = 72.88%
Semi-supervised la=5% (5/114 annotated): FlexTab AUROC = 85.97%, FlexTab-AD AUROC = 93.71%
Semi-supervised la=10% (11/114 annotated): FlexTab AUROC = 94.18%, FlexTab-AD AUROC = 96.76%
Semi-supervised la=50% (57/114 annotated): FlexTab AUROC = 99.23%, FlexTab-AD AUROC = 99.59%
Semi-supervised la=100% (114/114 annotated): FlexTab AUROC = 99.92%, FlexTab-AD AUROC = 99.82%

CLUSTERING
Hungarian Accuracy: 96.13%
ARI: 85.01%

MULTI-TABLE
...
FlexTab-Multi AUROC = 74.6% for Relbench dataset "rel-f1" task "driver-dnf"
```

### Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from flextab import FlexTabClassifier

# Load sample data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier, 8k context and 8-fold bagging gives best performance, reduce if running out of memory
clf = FlexTabClassifier(max_context_size=8192, bagging=8)

clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### Regression

```python
from sklearn.datasets import fetch_openml
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from flextab import FlexTabRegressor

# Load sample data
df = fetch_openml(data_id=531, as_frame=True)
X = df.data
y = df.target.astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize the regressor, 8k context and 8-fold bagging gives best performance, reduce if running out of memory
regressor = FlexTabRegressor(max_context_size=8192, bagging=8)

regressor.fit(X_train, y_train)

# Predict on the test set
predictions = regressor.predict(X_test)

r2 = r2_score(y_test, predictions)
print("R² Score:", r2)
```

### Entity matching

`FlexTabMatcher` predicts whether two aligned tables describe the same entity row by row. Provide left/right context tables together with a binary `match` label, then call `predict_matching` on aligned test pairs:

```python
from sklearn.metrics import accuracy_score, roc_auc_score

from flextab import FlexTabMatcher
from flextab.utils.test_utils import load_febrl4

left_train, left_test, right_train, right_test, y_train, y_test = load_febrl4()

matcher = FlexTabMatcher(max_context_size=8192, bagging=1)
matcher.fit(left_train, right_train, y_train)
predictions = matcher.predict(left_test, right_test)
print(f'Accuracy {accuracy_score(y_test, predictions):.2%}')
# Probabilities (e.g. for thresholding or AUROC):
# probas = matcher.predict_proba_matching(left_test, right_test)
# print(f'AUROC {roc_auc_score(y_test, probas[:, 1]):.2%}')
```

## Known Issues
No known issues

## How to obtain support
[Create an issue](https://github.com/SAP-samples/flextab/issues) in this repository if you find a bug or have questions about the content.

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2026 SAP SE or an SAP affiliate company. All rights reserved.

The source code in this project is released under the Apache License 2.0 (see [LICENSE](LICENSE)). The model checkpoints are released under the CC-BY-NC-4.0 license.
