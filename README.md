# Housing Prices Prediction

This project predicts **California housing prices** using multiple regression techniques and feature engineering.  
It demonstrates the full end-to-end data science workflow: **exploration → cleaning → feature engineering → modeling → evaluation → diagnostics**.  

---

## Workflow Overview

### Step 1. Setup & Imports
- Core libraries: **NumPy, Pandas, Matplotlib, Seaborn**
- Quality-of-life settings (pandas display, warnings ignored, styled plots)
- Dataset loaded from `data/housing.csv`

### Step 2. Exploratory Data Analysis
- Histograms for numeric features  
- Category distribution plots (`ocean_proximity`)  
- Correlation matrix & heatmaps  
- Scatterplots of **median house value vs median income** and geographic scatterplots (longitude, latitude, population, prices)  

### Step 3. Data Cleaning
- Missing values in `total_bedrooms` filled with **median imputation**  
- `ocean_proximity` transformed into **One-Hot Encoded variables**  
- Feature engineering:
  - `rooms_per_household`
  - `bedrooms_per_household`
  - `population_per_household`
- Dropped redundant raw columns (`population`, `total_rooms`, `households`)  

###  Step 4. Train/Test Split
- **Stratified sampling** by `income_cat` (to preserve income distribution)  
- Separate **features (X)** and **labels (y)**  

### Step 5. Baseline Models
- **Dummy Regressor** (predicts mean of target) → sanity check  
- **1-feature Linear Regression** using only `median_income`  

###  Step 6. Feature Scaling
- StandardScaler applied to **numeric features** only  
- Concatenated with categorical dummy variables  

### Step 7. Advanced Models
- **Multiple Linear Regression** (all features)  
- **Custom Stochastic Gradient Descent (SGD)** with batch updates, momentum-like decay, and convergence diagnostics  
- **k-Nearest Neighbors (kNN)** with:
  - GridSearchCV  
  - RandomizedSearchCV  
  - Bias–variance tradeoff plots  
- **Bagging with kNN** → ensemble approach for variance reduction  

### Step 8. Diagnostics & Evaluation
- Residual plots (distribution & vs predictions)  
- Coefficient interpretation (standardized regression weights)  
- Feature selection experiments (Top-1, Top-3, All features)  
- Metrics tracked:
  - RMSE (Root Mean Squared Error)  
  - MAE (Mean Absolute Error)  
  - R² (Coefficient of Determination)  

---

## Example Results

| Model                  | Test RMSE | Test MAE | Test R² |
|-------------------------|-----------|----------|---------|
| Dummy Regressor         | 114,165   | 90,807   | -0.0000 |
| Linear Regression (1f)  | 82,431    | 61,857   | 0.479   |
| Linear Regression (All) | 70,651    | 52,107   | 0.617   |
| Custom SGD              | 72,000    | 53,000   | 0.60    |
| kNN (best CV)           | 76,392    | 56,057   | 0.552   |
| Bagging kNN             | 76,428    | 56,072   | 0.552   |

---

## 🛠 Tech Stack

### Core Data Science
- **Pandas** → DataFrames, preprocessing, stratified sampling  
- **NumPy** → Numerical computation, correlation, gradients  
- **Matplotlib / Seaborn** → Visualization, scatterplots, heatmaps  

### Machine Learning
- **Scikit-learn**:
  - `train_test_split`, `StratifiedShuffleSplit`
  - `DummyRegressor`, `LinearRegression`
  - `StandardScaler`
  - `KNeighborsRegressor`, `GridSearchCV`, `RandomizedSearchCV`
  - `BaggingRegressor`
  - Metrics: `mean_squared_error`, `mean_absolute_error`, `r2_score`
- **Custom Implementations**:
  - Gradient Descent (batch SGD with convergence criteria, momentum-like updates)  

---

## Notes

- `Learning_Process.py` contains **all steps, experiments, and debugging notes**  
- `main.py` is a **clean, production-ready pipeline**  
- `Reports/` stores metrics in CSV/Markdown for easy comparison  
- `images/` stores plots for EDA, diagnostics, and model evaluation  

---

## Run the Project

```bash
git clone https://github.com/yourusername/housing-ml-project.git
cd housing-ml-project

# Run the main pipeline
python main.py
