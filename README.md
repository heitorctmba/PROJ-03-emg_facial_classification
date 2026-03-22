# EMG Signal Classification

Implementation and comparison of classification algorithms applied to electromyography (EMG) signals for facial expression recognition.

## Objective

Evaluate and compare different classification approaches on EMG sensor data, measuring accuracy and robustness across 100 random train/test splits (80/20).

## Data

| File | Description |
|---|---|
| `EMG_DATASET/EMG.csv` | 50,000 samples with 2 features (x1, x2) |
| `EMG_DATASET/Rotulos.csv` | 50,000 one-hot encoded labels across 5 classes |

**Classes:** neutro, sorriso, aberto, surpreso, grumpy

The dataset is perfectly balanced (10,000 samples per class) and contains no conflicting coordinates: no two samples share the same (x1, x2) values while belonging to different classes. 55% of the samples are exact duplicates within the same class, reducing the effective feature space to 22,523 unique coordinates.

## Notebook Structure

1. **Setup:** imports, helper functions, data loading
2. **Exploratory Data Analysis:** class distribution, descriptive statistics, scatter plot, box plot, quartile analysis, feature distributions, coordinate overlap analysis
3. **OLS (Ordinary Least Squares):** basic OLS, L2 regularization (Ridge), hyperplane visualization, effect of bias term
4. **Gaussian Bayesian Classifiers:** shared covariance, pooled covariance, Friedman LDA, Naive Bayes
5. **Results Comparison:** accuracy summary across all models
6. **Conclusion:** analysis and interpretation of results

## Key Results

| Model | Mean Accuracy (%) |
|---|---|
| OLS without bias | 40.17 |
| Basic OLS | 72.40 |
| Shared Covariance | 94.64 |
| Pooled Covariance | 96.22 |
| Friedman LDA | 96.26 |
| Naive Bayes | 99.23 |

Naive Bayes achieved the best result, explained by the low correlation between features x1 and x2 (-0.13) and the absence of intrinsic ambiguity in the data. L2 regularization had no effect on OLS accuracy across all tested lambda values. Removing the bias term from OLS dropped accuracy to ~40%, confirming its importance for positioning hyperplanes off the origin.

## Report

A detailed written report covering the methodology, mathematical background, and analysis of results is available in [`emg-facial-classification-report.pdf`](emg-facial-classification-report.pdf).

## Requirements

```
pip install -r requirements.txt
```

> Run Jupyter from the `av1/` directory or keep the workspace root at `sistemas_inteligentes/` as the notebook uses paths relative to the workspace.
