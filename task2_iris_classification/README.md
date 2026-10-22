# California Housing Price Prediction 

A machine learning regression project that predicts median house values using the California Housing dataset. The project covers data cleaning, exploratory data analysis, feature engineering, model training, cross-validation, and evaluation on unseen data.

##  Project Overview

The goal of this project is to build and compare regression models that estimate median house values based on housing and geographical features.

Three regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression

Linear Regression achieved the best results among the three models using the selected configurations.

##  Objectives

- Inspect and clean a real-world dataset.
- Handle missing values.
- Explore relationships between numerical features.
- Encode categorical variables and scale numerical features.
- Generate pairwise interaction features.
- Train and compare regression models.
- Evaluate model performance using cross-validation and test-set metrics.
- Visualize predictions and residuals.

##  Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Scikit-learn** — preprocessing, model training, cross-validation, and evaluation
- **Jupyter Notebook** — interactive development and experimentation

##  Dataset

The project uses the California Housing dataset provided as `housing.csv`.

The dataset contains **20,640 records and 10 columns**, including the target variable `median_house_value`.

The features include:

| Feature | Description |
|---|---|
| `longitude` | Geographic longitude |
| `latitude` | Geographic latitude |
| `housing_median_age` | Median age of housing units |
| `total_rooms` | Total rooms in the district |
| `total_bedrooms` | Total bedrooms in the district |
| `population` | District population |
| `households` | Number of households |
| `median_income` | Median household income |
| `ocean_proximity` | Categorical proximity to the ocean |
| `median_house_value` | Target variable to predict |

## ⚙️ Project Workflow

### 1. Data Inspection and Cleaning

- Inspected dataset dimensions, column types, and missing values.
- Identified 207 missing values in `total_bedrooms`.
- Imputed missing bedroom counts using the median.

### 2. Exploratory Data Analysis

- Generated descriptive statistics.
- Examined correlations between numerical variables.
- Created a correlation heatmap to explore feature relationships.

### 3. Feature Engineering and Preprocessing

- Separated input features (`X`) from the target (`y`).
- Split the dataset into training and testing sets using an 80/20 split.
- Used median imputation, pairwise interaction features, and standard scaling for numerical variables.
- Used most-frequent imputation and one-hot encoding for categorical variables.
- Applied preprocessing within a scikit-learn pipeline to prevent information leakage during cross-validation and model evaluation.

The resulting processed dataset contained 41 features.

### 4. Model Training

The following regression models were trained:

- **Linear Regression:** baseline regression model.
- **Ridge Regression:** linear regression with L2 regularization.
- **Lasso Regression:** linear regression with L1 regularization.

### 5. Model Evaluation

Five-fold cross-validation was used to compare model performance on the training data. The selected models were then evaluated on a separate held-out test set.

The evaluation metrics were:

- **RMSE:** Root Mean Squared Error; lower values indicate better performance.
- **MAE:** Mean Absolute Error; lower values indicate better performance.
- **R²:** Coefficient of determination; higher values generally indicate better explanatory performance.

### 6. Visualization

Two diagnostic visualizations were created:

- **Actual vs. Predicted Plot:** compares actual house values against model predictions.
- **Residual Plot:** examines prediction errors relative to predicted values.

##  Results

### Cross-Validation Results

| Model | Mean CV RMSE | Mean CV R² |
|---|---:|---:|
| Linear Regression | 64,621.885 | 0.687 |
| Ridge Regression | 65,505.692 | 0.679 |
| Lasso Regression | 65,980.326 | 0.674 |

### Test-Set Results

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| **Linear Regression** | **66,789.431** | **47,282.634** | **0.660** |
| Ridge Regression | 67,772.272 | 48,194.772 | 0.649 |
| Lasso Regression | 68,252.098 | 48,709.984 | 0.645 |

*Error metrics are expressed in the dataset's house-value units, which are US dollars.*

###  Best Model: Linear Regression

Linear Regression achieved the lowest test RMSE and MAE, along with the highest test R².

Its test R² of **0.660** indicates that the model explains approximately 66% of the variation in median house values in the held-out test set.

The results provide a useful baseline, although prediction errors remain substantial.

##  How to Run the Project

### Prerequisites

Install Python and the required libraries.

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/WAIRIADEV/california-housing-price-prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd california-housing-price-prediction
   ```

3. Place `housing.csv` in the project directory if it is not already included.

4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open the project's `.ipynb` notebook and run the cells in order.

> **Note:** Replace the example repository URL with your actual GitHub repository URL if you use a different repository name.

##  Future Improvements

- Experiment with Random Forest and Gradient Boosting regressors.
- Tune Ridge and Lasso regularization parameters.
- Compare models with and without interaction features.
- Investigate nonlinear relationships between housing features and prices.
- Examine prediction errors across different income levels and geographical areas.
- Investigate the effect of the dataset's capped median house values.
- Use a reproducible evaluation workflow and preserve the final fitted model.

##  Limitations

- The model's predictions have substantial errors.
- The dataset's target values are capped, which limits the representation of expensive properties.
- The model's performance may vary across geographical areas and market conditions.
- Results on this dataset do not guarantee performance on current real estate data.

##  Key Learnings

This project provided practical experience with:

- Data cleaning and missing-value imputation.
- Exploratory data analysis.
- Feature engineering and preprocessing pipelines.
- Regression algorithms and regularization.
- Cross-validation and model evaluation.
- Prediction and residual visualization.
- Building an end-to-end machine learning workflow.

##  Author

**Antony Wairia Kamau**

GitHub: [@WAIRIADEV](https://github.com/WAIRIADEV)

---


