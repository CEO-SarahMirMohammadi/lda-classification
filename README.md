# Linear Discriminant Analysis Classification

A machine learning project demonstrating **Linear Discriminant Analysis (LDA)** classification using a synthetic dataset containing **2,000 samples and 6 features**.

The project is based on Example 3.8 and modifies the original implementation to work with a larger dataset and a higher number of features.

## Overview

The original example uses:

* 1,000 samples
* 4 features

This modified version uses:

* **2,000 samples**
* **6 features**
* 2 informative features
* 0 redundant features

The workflow is:

```text
Generate Dataset
      ↓
Create LDA Model
      ↓
Train Model
      ↓
Make Prediction
```

## Algorithm

### Linear Discriminant Analysis

Linear Discriminant Analysis (LDA) is a supervised machine learning algorithm commonly used for classification and dimensionality reduction.

LDA attempts to find a linear combination of features that separates different classes.

In classification, the model learns the characteristics of each class from the training data and uses those learned characteristics to classify new observations.

## Dataset

This project uses Scikit-learn's `make_classification()` function to generate a synthetic classification dataset.

The dataset contains:

| Property             | Value |
| -------------------- | ----: |
| Samples              | 2,000 |
| Features             |     6 |
| Informative features |     2 |
| Redundant features   |     0 |
| Random state         |     0 |
| Shuffle              | False |

The generated feature matrix has the shape:

```text
(2000, 6)
```

Each sample therefore contains six numerical feature values.

## Dataset Generation

The dataset is generated using:

```python
X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False
)
```

### Parameters

#### `n_samples=2000`

Creates 2,000 observations.

#### `n_features=6`

Each observation contains six features.

#### `n_informative=2`

Two features contain information useful for separating the classes.

#### `n_redundant=0`

No redundant features are generated.

#### `random_state=0`

Ensures that the generated dataset is reproducible.

#### `shuffle=False`

Keeps the generated feature structure in its original order.

## Project Structure

```text
lda-classification/
│
├── lda_classification.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

| Requirement  | Purpose                                   |
| ------------ | ----------------------------------------- |
| Python 3.x   | Programming language                      |
| scikit-learn | Dataset generation and LDA implementation |

## Installation

Clone the repository:

```bash
git clone https://github.com/CEO-SarahMirMohammadi/lda-classification.git
```

Move into the project directory:

```bash
cd lda-classification
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Program

Run the Python program:

```powershell
python lda_classification.py
```

The program will:

1. Generate 2,000 samples.
2. Generate 6 features for each sample.
3. Display the generated dataset.
4. Display the dataset shape.
5. Create an LDA classifier.
6. Train the classifier.
7. Predict the class of a new six-feature sample.

## Prediction

The program makes a prediction for:

```text
[0, 0, 0, 0, 0, 0]
```

The sample contains exactly six feature values, matching the six features used to train the model.

The output will be a class label such as:

```text
Prediction:
[1]
```

The exact predicted class is determined by the generated dataset and trained LDA model.

## Workflow

```text
             make_classification()
                     │
                     ▼
          2,000 samples × 6 features
                     │
                     ▼
                 X and y
                     │
                     ▼
          LinearDiscriminantAnalysis
                     │
                     ▼
                 clf.fit()
                     │
                     ▼
             Trained LDA Model
                     │
                     ▼
          New 6-Feature Observation
                     │
                     ▼
              clf.predict()
                     │
                     ▼
               Class Prediction
```

## Key Concepts

### 1. Supervised Learning

LDA is a supervised learning algorithm because the training dataset contains both:

* Input features `X`
* Target labels `y`

### 2. Linear Discriminant Analysis

LDA finds a linear combination of features that helps separate different classes.

### 3. Feature Matrix

The feature matrix `X` contains 2,000 rows and 6 columns:

```python
X.shape
```

Output:

```text
(2000, 6)
```

### 4. Target Vector

The `y` array contains the class label associated with each sample.

### 5. Model Training

The LDA model is trained using:

```python
clf.fit(X, y)
```

### 6. Prediction

After training, the model can classify a new observation:

```python
clf.predict([[0, 0, 0, 0, 0, 0]])
```

The input must contain six features because the model was trained with six features.

## Why Use Synthetic Data?

Synthetic datasets are useful for learning and testing machine learning algorithms because they allow us to control important characteristics of the dataset.

For example, we can specify:

* Number of samples
* Number of features
* Number of informative features
* Number of redundant features
* Random seed

This makes `make_classification()` useful for experimenting with machine learning algorithms.

## Output

A typical execution will begin with output similar to:

```text
Dataset:
[[... ... ... ... ... ...]
 [... ... ... ... ... ...]
 [... ... ... ... ... ...]
 ...
]

Dataset shape:
(2000, 6)

Model trained successfully.

Prediction:
[...]
```

The complete dataset is printed by the program, so the output will contain 2,000 rows.

## Future Improvements

Possible improvements include:

* Add a train/test split.
* Calculate classification accuracy.
* Generate a confusion matrix.
* Generate a classification report.
* Visualize the synthetic dataset.
* Compare LDA with Logistic Regression.
* Compare LDA with Naive Bayes.
* Experiment with different numbers of informative features.
* Experiment with different sample sizes.
* Add cross-validation.
* Save the trained model using Joblib.
* Create a prediction API using FastAPI.

## License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and build upon this project under the terms of the MIT License.