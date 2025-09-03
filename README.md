# 🏠 Housing Prices Prediction

This project builds and evaluates machine learning models to predict **California housing prices** based on demographic and geographic features.  
It combines **experimentation (Learning Process)** with a **clean main pipeline** for reproducibility.

---

## 🚀 Quickstart

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/housing-ml-project.git
   cd housing-ml-project
   ```

2. **Add dataset**
   Place your dataset in the `data/` folder:
   ```
   data/housing.csv
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the pipeline**
   ```bash
   python main.py
   ```

---

## 📊 Features

- Data preprocessing & exploration  
- Multiple regression models:
  - Dummy baseline
  - Linear Regression (Scikit-learn)
  - Custom Linear Regression (Gradient Descent)
  - k-Nearest Neighbors (with cross-validation)
  - Bagging kNN (variance reduction)
- Automated reporting of metrics (`Reports/metrics.csv`, `Reports/metrics.md`)  
- Visualization (`images/`) including:
  - Correlation heatmap
  - Target distribution
  - Income vs house value scatter
  - Geographic price scatter
  - Predictions vs actuals
  - Residuals analysis  

---

## 📈 Example Output

| Model                  | Test RMSE | Test R² |
|-------------------------|-----------|---------|
| Linear Regression       | 71,540    | 0.61    |
| kNN (best CV)           | 61,221    | 0.72    |
| Bagged kNN              | 58,742    | 0.75    |
| Custom Linear (GD)      | 72,822    | 0.60    |
| Dummy Regressor (mean)  | 118,980   | -0.02   |

---

## 🛠 Tech Stack

- **Python 3.9+**
- **Pandas / NumPy** → Data handling  
- **Scikit-learn** → ML models, metrics, pipelines  
- **Matplotlib / Seaborn** → Visualizations  

---

## 📌 Notes

- `Learning_Process.py` documents experimentation, debugging, and step-by-step learning.
- `main.py` is the production-ready pipeline.
