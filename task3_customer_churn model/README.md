# Telecom Customer Churn Prediction

## Project Overview

This project uses machine learning to predict customer churn from telecom subscription records. The objective is to identify customers who are likely to discontinue their services, evaluate ensemble classification models, and investigate the features most influential in their predictions.

Two ensemble models were trained and evaluated: **Random Forest** and **XGBoost**. Both achieved a test ROC-AUC above 0.80.

## Business Problem

Customer churn can reduce recurring revenue and increase customer acquisition costs. By identifying customers at higher risk of leaving, telecom providers can investigate potential service issues and develop targeted customer retention strategies.

This project explores how customer tenure, contract type, billing information, internet service, and optional support services relate to churn predictions.

## Dataset

- **Source:** IBM Telco Customer Churn dataset
- **Original records:** 7,043
- **Records after cleaning:** 7,032
- **Original columns:** 21
- **Columns after cleaning:** 20, including the target variable
- **Target variable:** `Churn` (`Yes` or `No`)

### Data preparation

The following steps were performed:

1. Converted `TotalCharges` to numeric values.
2. Identified 11 missing `TotalCharges` values and removed those records after inspection.
3. Removed `customerID`, which is an identifier rather than a useful predictive feature.
4. Separated predictors from the target variable.
5. Identified four numeric features and 15 categorical features.
6. Split the dataset into 80% training data and 20% testing data using stratified sampling.

The final dataset contained 7,032 records with no missing values.

## Exploratory Data Analysis

The cleaned dataset contained:

| Churn status | Customers | Percentage |
|---|---:|---:|
| Stayed | 5,163 | 73.42% |
| Churned | 1,869 | 26.58% |

Contract type showed a strong association with churn:

| Contract type | Churn rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.28% |
| Two year | 2.85% |

These findings suggest that contract type is an important predictor to investigate. The observed relationships do not, by themselves, establish causation.

## Technologies Used

- Python
- pandas and NumPy
- Matplotlib
- scikit-learn
- XGBoost
- JupyterLab

## Methodology

### Preprocessing

A scikit-learn `ColumnTransformer` was used to apply separate preprocessing to numeric and categorical features.

- **Numeric features:** median imputation and standardization.
- **Categorical features:** most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`.

Preprocessing was fitted within the model pipelines to prevent information leakage from the held-out test set.

### Class imbalance

The target distribution was imbalanced, with approximately 26.58% of customers churning.

Class weighting was used as the initial strategy:

- Random Forest: `class_weight="balanced"`
- XGBoost: `scale_pos_weight` calculated from the training target distribution.

SMOTE was not used in this experiment.

### Models

**Random Forest Classifier**

A bagging-based ensemble that combines predictions from multiple decision trees.

**XGBoost Classifier**

A gradient-boosted tree model trained sequentially to improve predictive performance.

Both models were evaluated using five-fold stratified cross-validation on the training set. Final performance was measured on the held-out test set.

## Model Evaluation

### Cross-validation results

| Metric | XGBoost | Random Forest |
|---|---:|---:|
| Mean ROC-AUC | 0.8453 | 0.8254 |
| ROC-AUC standard deviation | 0.0051 | 0.0070 |
| Mean accuracy | 0.7628 | 0.7756 |
| Mean precision | 0.5380 | 0.5687 |
| Mean recall | 0.7652 | 0.6441 |

### Test set results

| Metric | XGBoost | Random Forest |
|---|---:|---:|
| ROC-AUC | **0.8294** | 0.8154 |
| Accuracy | 73.56% | **76.05%** |
| Precision | 50.17% | **54.14%** |
| Recall | **77.81%** | 64.71% |
| F1-score | **0.6101** | 0.5895 |

Both models exceeded the project's target test ROC-AUC of 0.80.

### Interpretation

XGBoost achieved the stronger ROC-AUC, indicating better ranking performance across classification thresholds. It also achieved higher recall, identifying a larger proportion of actual churners at the default classification threshold.

Random Forest achieved higher accuracy and precision, meaning its positive churn predictions were more reliable proportionally, although it missed more actual churners.

The preferred model depends on business priorities. If identifying as many potential churners as possible is the main objective, XGBoost is the stronger starting point. The classification threshold should be tuned according to retention costs and the consequences of false positives and false negatives.

## Feature Importance

### XGBoost

The leading features included:

| Feature | Importance |
|---|---:|
| Month-to-month contract | 0.3061 |
| Fiber optic internet | 0.0858 |
| No online security | 0.0570 |
| No internet service — online backup | 0.0532 |
| No online security service | 0.0343 |

### Random Forest

The leading features included:

| Feature | Importance |
|---|---:|
| Total charges | 0.1410 |
| Tenure | 0.1320 |
| Monthly charges | 0.1219 |
| Month-to-month contract | 0.0765 |
| No online security | 0.0378 |

XGBoost assigned a relatively high importance score to month-to-month contracts. Random Forest distributed importance more broadly across charges, tenure, and contract-related features.

These values represent model-specific feature-importance scores, not causal effects. Further analysis using permutation importance or SHAP could help investigate the stability and direction of individual feature contributions.

## Business Recommendations

Based on the observed patterns, a telecom provider could:

1. Investigate why month-to-month customers churn at a higher rate.
2. Evaluate whether customers using fiber optic services report distinct satisfaction or service-quality issues.
3. Examine the relationship between support-service subscriptions and churn.
4. Test targeted retention campaigns for customers identified as high risk.
5. Measure campaign effectiveness through controlled experiments rather than assuming that predictive associations imply successful interventions.

## Limitations

- The dataset represents a particular telecom customer population and may not generalize to other providers.
- Feature importance does not establish causation.
- The default classification threshold may not be optimal for business use.
- No external validation dataset was used.
- Class weighting was evaluated, but SMOTE and threshold optimization were not compared.
- ROC-AUC does not directly measure the financial value of retention actions.

## Conclusion

This project demonstrates an end-to-end machine learning classification workflow, from data cleaning and exploratory analysis through preprocessing, cross-validation, model comparison, test evaluation, and feature-importance analysis.

Both models exceeded the target test ROC-AUC of 0.80. XGBoost achieved the best test ROC-AUC of **0.8294**, while Random Forest achieved **0.8154**.

The results show how ensemble machine learning can support customer churn analysis and provide a foundation for future work in threshold optimization, model explainability, and retention strategy evaluation.

## Future Improvements

- Compare class weighting against SMOTE using an imbalanced-learn pipeline.
- Tune model hyperparameters using cross-validation.
- Optimize classification thresholds using validation data.
- Add precision-recall curves and calibration analysis.
- Use SHAP or permutation importance for deeper model interpretation.
- Evaluate potential business value using retention costs and campaign outcomes.

## Author

**Antony Wairia Kamau**

