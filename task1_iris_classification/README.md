# Task 1: Multi-Class Benchmark Classification Pipeline (Iris)

## Objective
Build an end-to-end supervised classification pipeline on the Iris dataset: scale the features, split the data 80/20, train Logistic Regression and a Decision Tree, and benchmark accuracy, precision, recall and F1-score.

## Files
| File | Description |
|------|-------------|
| `iris_classification.ipynb` | Full notebook: code, outputs, plots and written conclusions |
| `iris.csv` | Iris dataset (150 samples, 4 features, 3 species) |

## Approach
1. Load the data with Pandas and check shape, missing values and class balance.
2. Explore the data with a Seaborn pairplot.
3. Split the data 80/20 (stratified), then scale with `StandardScaler`, fitting on the training set only.
4. Train `LogisticRegression` and `DecisionTreeClassifier` (`max_depth=3`).
5. Evaluate with accuracy, macro precision, macro recall, macro F1 and classification reports.
6. Plot confusion matrix heatmaps for both models.
7. Plot decision boundaries using the two petal features.

## Results (test set, 30 samples)
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Decision Tree | 0.9667 | 0.9697 | 0.9667 | 0.9666 |

## Key findings
- Setosa is classified perfectly; all mistakes are between versicolor and virginica.
- Petal features separate the species better than sepal features.
- The Decision Tree scored one sample higher, which is not conclusive on a 30-sample test set.
- Logistic Regression draws slanted linear boundaries; the Decision Tree draws axis-aligned rectangles.

## How to run
```bash
pip install -r ../requirements.txt
jupyter notebook iris_classification.ipynb
```
The notebook reads `iris.csv` from the same folder, so run it from inside this folder.

## Notes
The notebook ends with a section on the errors I hit while building the pipeline and how I fixed each one.
