import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Tuple, Dict

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint

# ------------------------- Paths -------------------------
DATA = Path("data") / "housing.csv"
IMAGES = Path("images")
REPORTS = Path("reports")
IMAGES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ------------------------- Utils -------------------------
def eval_metrics(y_true, y_pred) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def standardize_numeric(train_df: pd.DataFrame, test_df: pd.DataFrame):
    num_cols = train_df.select_dtypes(include=np.number).columns
    cat_cols = train_df.columns.difference(num_cols)

    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(train_df[num_cols]),
        columns=num_cols, index=train_df.index
    )
    X_test_num = pd.DataFrame(
        scaler.transform(test_df[num_cols]),
        columns=num_cols, index=test_df.index
    )

    X_train_cat = train_df[cat_cols].copy()
    X_test_cat  = test_df[cat_cols].copy()

    X_train = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test  = pd.concat([X_test_num, X_test_cat], axis=1)
    return X_train, X_test

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"total_rooms", "households"}.issubset(out.columns):
        out["rooms_per_household"] = out["total_rooms"] / out["households"]
    if {"total_bedrooms", "households"}.issubset(out.columns):
        out["bedrooms_per_household"] = out["total_bedrooms"] / out["households"]
    if {"population", "households"}.issubset(out.columns):
        out["population_per_household"] = out["population"] / out["households"]
    # Drop raw cols that were decomposed (optional; keeps feature count modest)
    drop_cols = [c for c in ["population", "total_rooms", "total_bedrooms", "households"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out

def ensure_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=False)
    return df

# ------------------------- Mini-batch GD for Linear -------------------------
def mb_gd_fit_predict(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, *, features: list,
    lr: float = 0.05, batch_size: int = 256, n_iter: int = 100,
    tol: float = 1e-4, random_state: int = 42
) -> np.ndarray:
    """
    Trains a linear model via mini-batch GD **on standardized features and target**.
    Returns predictions on X_test in the ORIGINAL target scale.
    """
    rng = np.random.default_rng(seed=random_state)

    # Work on chosen features only
    Xtr = X_train[features].copy()
    Xte = X_test[features].copy()

    # Standardize features and target
    xs = StandardScaler()
    ys = StandardScaler()

    Xtr_s = xs.fit_transform(Xtr.values)
    Xte_s = xs.transform(Xte.values)

    ytr_s = ys.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Add intercept
    Xtr_s = np.c_[np.ones((Xtr_s.shape[0], 1)), Xtr_s]
    Xte_s = np.c_[np.ones((Xte_s.shape[0], 1)), Xte_s]

    n_obs, n_vars = Xtr_s.shape
    theta = rng.normal(size=n_vars)

    def grad(xb, yb, th):
        m = xb.shape[0]
        return (2.0/m) * xb.T.dot(xb.dot(th) - yb)

    xy = np.c_[Xtr_s, ytr_s]
    last_update = np.zeros_like(theta)

    for epoch in range(n_iter):
        rng.shuffle(xy)
        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            xb, yb = xy[start:stop, :-1], xy[start:stop, -1]
            g = grad(xb, yb, theta)
            update = -lr * g
            theta += update
            last_update = update
        if np.all(np.abs(last_update) <= tol):
            break

    # Predict on test (standardized space), inverse-transform to original y scale
    yte_pred_s = Xte_s.dot(theta)
    yte_pred = ys.inverse_transform(yte_pred_s.reshape(-1, 1)).ravel()
    return yte_pred

# ------------------------- Main -------------------------
def main():
    df = pd.read_csv(DATA)

    # Basic cleaning
    if "total_bedrooms" in df.columns:
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

    df = ensure_one_hot(df)
    df = add_engineered_features(df)

    target = "median_house_value"

    # Stratified split by binned median_income (if present), else plain split
    if "median_income" in df.columns:
        df["income_cat"] = pd.cut(
            df["median_income"],
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
            labels=[1,2,3,4,5]
        )
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in splitter.split(df, df["income_cat"]):
            train_df = df.loc[train_idx].drop(columns=["income_cat"])
            test_df  = df.loc[test_idx].drop(columns=["income_cat"])
    else:
        # Fallback if the column is missing
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].copy()
    X_test  = test_df.drop(columns=[target])
    y_test  = test_df[target].copy()

    # Standardize numeric features only (one-hot kept as-is)
    X_train, X_test = standardize_numeric(X_train, X_test)

    # -------------------- Models --------------------
    results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    # 1) Dummy baseline
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    results["Dummy"] = {
        "Train": eval_metrics(y_train, dummy.predict(X_train)),
        "Test":  eval_metrics(y_test,  dummy.predict(X_test)),
    }

    # 2) Linear Regression (sklearn) on all features
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    ytr_lin = lin.predict(X_train)
    yte_lin = lin.predict(X_test)
    results["Linear (sklearn)"] = {
        "Train": eval_metrics(y_train, ytr_lin),
        "Test":  eval_metrics(y_test,  yte_lin),
    }

    # 3) Linear via your GD (same feature subset)
    # Use top-3 correlated features with y on training set
    corrs = []
    for c in X_train.columns:
        try:
            corrs.append((c, float(np.corrcoef(X_train[c], y_train)[0,1])))
        except Exception:
            pass
    corrs = sorted([(c, abs(v)) for c, v in corrs if not np.isnan(v)], key=lambda t: t[1], reverse=True)
    feat_subset = [c for c, _ in corrs[:3]] if len(corrs) >= 3 else list(X_train.columns[:3])
    yte_gd = mb_gd_fit_predict(X_train, y_train, X_test, features=feat_subset, lr=0.05, batch_size=256, n_iter=200)
    # For completeness, compute train preds for GD via refit on full train and predict on train too
    # (re-using same function but X_test = X_train)
    ytr_gd = mb_gd_fit_predict(X_train, y_train, X_train, features=feat_subset, lr=0.05, batch_size=256, n_iter=200)
    results[f"Linear via GD ({', '.join(feat_subset)})"] = {
        "Train": eval_metrics(y_train, ytr_gd),
        "Test":  eval_metrics(y_test,  yte_gd),
    }

    # 4) kNN (best from CV) on a small subset to keep runtime reasonable
    knn_features = feat_subset[:2] if len(feat_subset) >= 2 else feat_subset
    Xtr_knn = X_train[knn_features]
    Xte_knn = X_test[knn_features]

    grid = {
        "n_neighbors": list(range(50, 201, 10)),
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }
    gs = GridSearchCV(
        KNeighborsRegressor(),
        grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    gs.fit(Xtr_knn, y_train)
    best_knn = gs.best_estimator_
    ytr_knn = best_knn.predict(Xtr_knn)
    yte_knn = best_knn.predict(Xte_knn)
    results["kNN (best CV)"] = {
        "Train": eval_metrics(y_train, ytr_knn),
        "Test":  eval_metrics(y_test,  yte_knn),
    }

    # 5) Bagged kNN
    # scikit-learn >=1.2 uses 'estimator'; for older versions, fall back to 'base_estimator'
    try:
        bag_knn = BaggingRegressor(estimator=best_knn, n_estimators=100, bootstrap=True, n_jobs=-1, random_state=42)
    except TypeError:
        bag_knn = BaggingRegressor(base_estimator=best_knn, n_estimators=100, bootstrap=True, n_jobs=-1, random_state=42)
    bag_knn.fit(Xtr_knn, y_train)
    ytr_bag = bag_knn.predict(Xtr_knn)
    yte_bag = bag_knn.predict(Xte_knn)
    results["Bagged kNN"] = {
        "Train": eval_metrics(y_train, ytr_bag),
        "Test":  eval_metrics(y_test,  yte_bag),
    }

    # -------------------- Save comparison table --------------------
    rows = []
    for model, scores in results.items():
        tr = scores["Train"]; te = scores["Test"]
        rows.append({
            "Model": model,
            "Train_RMSE": tr[0], "Train_MAE": tr[1], "Train_R2": tr[2],
            "Test_RMSE": te[0],  "Test_MAE": te[1],  "Test_R2": te[2],
        })
    comp = pd.DataFrame(rows).sort_values("Test_R2", ascending=False).reset_index(drop=True)
    comp.to_csv(REPORTS / "metrics.csv", index=False)

    # Also save as Markdown
    md_lines = ["| Model | Train RMSE | Train MAE | Train R² | Test RMSE | Test MAE | Test R² |",
                "|------|-----------:|----------:|---------:|----------:|---------:|--------:|"]
    for _, r in comp.iterrows():
        md_lines.append(
            f"| {r['Model']} | {r['Train_RMSE']:.2f} | {r['Train_MAE']:.2f} | {r['Train_R2']:.4f} | "
            f"{r['Test_RMSE']:.2f} | {r['Test_MAE']:.2f} | {r['Test_R2']:.4f} |"
        )
    (REPORTS / "metrics.md").write_text("\n".join(md_lines), encoding="utf-8")

    # -------------------- Plots (matplotlib only) --------------------
    # Pick top 2 models by Test R2 for the predictions vs actual plot
    top2 = comp.nlargest(2, "Test_R2")["Model"].tolist()
    preds = {}
    for m in top2:
        if m == "Linear (sklearn)":
            preds[m] = yte_lin
        elif m.startswith("Linear via GD"):
            preds[m] = yte_gd
        elif m == "kNN (best CV)":
            preds[m] = yte_knn
        elif m == "Bagged kNN":
            preds[m] = yte_bag
        elif m == "Dummy":
            preds[m] = dummy.predict(X_test)

    # 1) Predictions vs Actual (scatter) for top 2
    plt.figure()
    for m in top2:
        plt.scatter(y_test, preds[m], alpha=0.5, label=m)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predictions vs Actual (Top 2 Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES / "pred_vs_actual_top2.png", dpi=200)
    plt.close()

    # 2) Residual distribution comparison: Linear vs kNN
    # Choose linear + the best kNN variant available
    linear_res = y_test - yte_lin
    # prefer bagged if present in results; else knn
    knn_name = "Bagged kNN" if "Bagged kNN" in results else "kNN (best CV)"
    knn_res = y_test - (yte_bag if knn_name == "Bagged kNN" else yte_knn)

    plt.figure()
    # histograms on same axes
    plt.hist(linear_res, bins=40, alpha=0.6, label="Linear (sklearn)", density=True)
    plt.hist(knn_res, bins=40, alpha=0.6, label=knn_name, density=True)
    plt.xlabel("Residual (y - ŷ)")
    plt.ylabel("Density")
    plt.title("Residual Distributions: Linear vs kNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES / "residuals_linear_vs_knn.png", dpi=200)
    plt.close()


        # -------------------- Extra Plots --------------------
    # 3) Correlation heatmap
    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(IMAGES / "correlation_heatmap.png", dpi=200)
    plt.close()

    # 4) Target distribution
    plt.figure()
    sns.histplot(df[target], bins=40, kde=True)
    plt.xlabel("Median House Value")
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(IMAGES / "target_distribution.png", dpi=200)
    plt.close()

    # 5) Scatter vs. median_income
    plt.figure()
    sns.scatterplot(x=df["median_income"], y=df[target], alpha=0.3)
    plt.xlabel("Median Income")
    plt.ylabel("Median House Value")
    plt.title("Target vs Median Income")
    plt.tight_layout()
    plt.savefig(IMAGES / "scatter_income_vs_target.png", dpi=200)
    plt.close()

    # 6) Geographic scatter (location, population, price)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        df["longitude"], df["latitude"],
        c=df[target], cmap="viridis",
        s=(df["population"]/100), alpha=0.4
    )
    plt.colorbar(scatter, label="Median House Value")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Location, Population, and Price")
    plt.tight_layout()
    plt.savefig(IMAGES / "geo_scatter.png", dpi=200)
    plt.close()

    # 7) Residuals vs Predictions for all models
    plt.figure()
    for m, preds_m in preds.items():
        residuals = y_test - preds_m
        plt.scatter(preds_m, residuals, alpha=0.5, label=m)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals (y - ŷ)")
    plt.title("Residuals vs Predictions (Top Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES / "residuals_vs_predictions.png", dpi=200)
    plt.close()


    # -------------------- Console summary --------------------
    print("\n=== Model Comparison (sorted by Test R²) ===")
    print(comp.to_string(index=False))
    print(f"\nSaved: {REPORTS / 'metrics.csv'}")
    print(f"Saved: {REPORTS / 'metrics.md'}")
    print(f"Saved: {IMAGES / 'pred_vs_actual_top2.png'}")
    print(f"Saved: {IMAGES / 'residuals_linear_vs_knn.png'}")

if __name__ == "__main__":
    main()
