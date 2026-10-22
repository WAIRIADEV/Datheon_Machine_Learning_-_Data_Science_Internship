# Datheon Machine Learning & Data Science Internship

Submission by **Antony Wairia** ([@WAIRIADEV](https://github.com/WAIRIADEV)).

This repository contains my solutions to the Datheon Machine Learning & Data Science internship tasks. Each task lives in its own folder with its code, data, requirements and notes.

## Progress

| # | Task | Status | Folder |
|---|------|--------|--------|
| 1 | Multi-Class Benchmark Classification Pipeline (Iris) | Done | [`task1_iris_classification/`](task1_iris_classification/) |

More tasks will be added as I complete them.

## Repository structure

```
.
├── README.md
└── task1_iris_classification/
    ├── README.md
    ├── iris_classification.ipynb
    ├── iris.csv
    └── requirements.txt
```

## Task 1: Multi-Class Benchmark Classification Pipeline (Iris)

An end-to-end supervised classification pipeline on the Iris dataset (150 samples, 4 features, 3 balanced species).

**Steps**
1. Loaded and inspected the data with Pandas.
2. Explored it with a Seaborn pairplot.
3. Split the data 80/20 (stratified), then scaled with `StandardScaler`, fitting on the training set only to avoid data leakage.
4. Trained Logistic Regression and a Decision Tree (`max_depth=3`).
5. Benchmarked accuracy, macro precision, macro recall and macro F1.
6. Plotted confusion matrix heatmaps for both models.
7. Plotted decision boundaries using the two petal features.

**Results (test set, 30 samples)**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree | 0.9667 | 0.9697 | 0.9667 | 0.9666 |

**Key findings**
- Setosa is classified perfectly by both models. All errors are between versicolor and virginica.
- Petal features separate the species better than sepal features.
- The Decision Tree scored one sample higher (29/30 vs 28/30), which is not conclusive on a test set this small.
- Logistic Regression draws slanted linear boundaries, while the Decision Tree draws axis-aligned rectangles.

The notebook also documents the errors I ran into and how I fixed each one.

## How to run

```bash
git clone https://github.com/WAIRIADEV/Datheon_Machine_Learning_-_Data_Science_Internship.git
cd Datheon_Machine_Learning_-_Data_Science_Internship/task1_iris_classification
pip install -r requirements.txt
jupyter notebook iris_classification.ipynb
```

## Tools

Python, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Jupyter Notebook.

## Dataset

Iris dataset from the [seaborn-data repository](https://github.com/mwaskom/seaborn-data).
