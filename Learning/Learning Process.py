# Core data science stack
from matplotlib.pylab import randint
import pandas as pd      # tables/dataframes
import numpy as np       # numerical ops
from pandas.plotting import scatter_matrix # scatter matrix plot let u visualize relationships in one go
# Visualization
import matplotlib.pyplot as plt  # plotting
import seaborn as sns          # nicer statistical plots
#---------------- STEP1 Setup ----------------

# Quality of life: wider tables, fewer warnings
pd.set_option("display.max_columns", 50)   # show more columns before truncating
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")  # pretty floats
pd.set_option("display.max_rows", 100)     # show up to 100 rows
pd.set_option("display.width", 1000)       # widen the console output
pd.set_option("display.precision", 3)      # round floats to 3 decimals

import warnings; warnings.filterwarnings("ignore")  # keep the output clean

# Matplotlib defaults (bigger, readable)
plt.rcParams["figure.figsize"] = (9, 6)   # default figure size
plt.rcParams["axes.grid"] = True          # light grids help readability

# Seaborn theme
sns.set_theme(context="notebook", style="whitegrid")  # pleasant default style
df = pd.read_csv("data/housing.csv")

#---------------- STEP2 Understand your Data ----------------
# Select only numeric columns for histograms
num_cols = df.select_dtypes(include=[np.number]).columns

# Many ML issues show up in distributions (skew, outliers, weird scales)
# df[num_cols].hist(bins=40, figsize=(14, 10), layout=(len(num_cols)//3+1, 3))
# plt.suptitle("Numeric Feature Distributions", y=1.02)
# plt.tight_layout() # Adjust layout to prevent overlap
# plt.show()

# df["ocean_proximity"].value_counts().plot(kind="bar")
# plt.title("Ocean Proximity Value Counts")
# plt.xlabel("Ocean Proximity")
# plt.ylabel("Counts")
# plt.xticks(rotation=45) # rotate x labels
# plt.tight_layout()
# plt.show()

# ---------------- STEP3 Understand Relationships and see which are features and targets ----------------
# Correlation matrix (numeric columns only)
target = "median_house_value"
# corr = df.select_dtypes(include=[np.number]).corr()
# '''computes the pairwise correlation coefficients between all numeric columns.
# By default, this is Pearson’s correlation (linear relationship, ranging from -1 to 1).
# Result: a square correlation matrix (DataFrame) where:
# Diagonal = 1.0 (correlation with itself).
# Symmetric: corr[A][B] = corr[B][A].'''


# # Heatmap to see relationships at a glance
# plt.figure(figsize=(10, 8)) # makes a new figure with width = 10, height = 8 (inches). Larger size helps readability.
# sns.heatmap(corr[[target]].sort_values(by=target, ascending=False), annot=False, cmap="crest", center=0)
# plt.title("Correlation Heatmap (Numeric Features)")
# plt.show()

# ---------------- STEP4  Scatter: target vs. a few strong predictors ----------------
# Helper to make consistent scatterplots vs. target
def scatter_vs_target(xcol, ycol=target, alpha=0.3):
    sns.scatterplot(x=df[xcol], y=df[ycol], alpha=alpha)
    plt.title(f"{ycol} vs {xcol}")
    plt.xlabel(xcol); plt.ylabel(ycol)
    plt.show()
# scatter_vs_target("median_income")


# A classic plot for this dataset: location matters!

# scatter = plt.scatter(
#     df["longitude"], df["latitude"],
#     c=df["median_house_value"],      # color encodes price
#     s=(df["population"] / 100),      # point size encodes population
#     cmap="viridis",                  # colormap
#     alpha=0.4
# )

# plt.colorbar(scatter, label="Median House Value")  # colorbar with label
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title("Location, Population, and Price")
# plt.show()

#Using scatter_matrix to visualize relationships
# scatter_matrix(df[["median_house_value", "median_income", "total_rooms", "df_median_age"]], figsize=(12, 8), diagonal="kde")
# plt.suptitle("Scatter Matrix of New Features")
# plt.show()

# ---------------- STEP4  Cleaning ----------------
'''We can either drop rows or fill them with something (e.g., median). Median is safer. Median works better than mean for skewed distributions.'''
# print(f"Missing values in 'total_bedrooms': {df['total_bedrooms'].isnull().sum()} out of {len(df)}")
median_bedrooms = df["total_bedrooms"].median()
df["total_bedrooms"].fillna(median_bedrooms, inplace=True)
# print(f"Missing values in 'total_bedrooms': {df['total_bedrooms'].isnull().sum()} out of {len(df)}")

# Turn Categorical into One-Hot for machine learning to understand 
df = pd.get_dummies(df, columns=["ocean_proximity"])
# 1. Create rooms_per_household
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
df["population_per_household"] = df["population"] / df["households"]
df = df.drop(["population", "total_rooms", "total_bedrooms", "households"], axis=1)

# ----------------- STEP5 Split train/test data -----------------
#Alternative: scikit-learn’s train_test_split has a stratify= parameter:
# train, test = train_test_split(df, test_size=0.2, stratify=df['income_cat'], random_state=42)
# however this is not as flexible as the following but it does the job
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# 1. Create an income category attribute for stratification
df["income_cat"] = pd.cut(
    df["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],   # bins for income ranges
    labels=[1, 2, 3, 4, 5]                  # assign category labels
)
# 2. Stratified split based on the income category
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
'''When you split your dataset (train_test_split), the rows are shuffled randomly before splitting.
random_state is just the seed for the random number generator.
Setting it ensures the same shuffle every time you run the code, so you get reproducible results.'''
for train_idx, test_idx in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_idx]
    strat_test_set = df.loc[test_idx]

# 3. Remove the income_cat column (it was just temporary)
for _set in (strat_train_set, strat_test_set):
    _set.drop("income_cat", axis=1, inplace=True)

# 4. Separate features (X) and labels (y)
X_train = strat_train_set.drop("median_house_value", axis=1)
y_train = strat_train_set["median_house_value"].copy()
'''Without .copy(), y_train is just a view (reference) of the original train_set["median_house_value"].
That means if you later modify y_train, it might also modify the original train_set column (and sometimes vice versa).
so .copy makes this totally separable, SAME for the drop()'''
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()



# ----------------- STEP6 Baseline 1: Dummy Regressor -----------------
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 1) Fit Dummy (predicts mean of y_train)
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)

# 2) Predict
y_train_pred = dummy.predict(X_train)
y_test_pred = dummy.predict(X_test)

# 3) Evaluate
def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

dummy_train = eval_metrics(y_train, y_train_pred)
dummy_test = eval_metrics(y_test, y_test_pred)
# Dummy Train: 115697.79364533882 91296.46145607694 0.0 
# Dummy Test : 114165.27031417929 90806.78845743735 -4.2875059214297906e-05
from sklearn.linear_model import LinearRegression

# Baseline 2: 1-feature Linear Regression (median_income)
# 1) Use only one feature: median_income
X_train_mi = X_train[["median_income"]]
X_test_mi  = X_test[["median_income"]]

# 2) Fit linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_mi, y_train)

# 3) Predict
y_train_pred = lin_reg.predict(X_train_mi)
y_test_pred  = lin_reg.predict(X_test_mi)

# 4) Evaluate
lin_train_1f = eval_metrics(y_train, y_train_pred)
lin_test_1f = eval_metrics(y_test, y_test_pred)

# 1-Feature Linear Train: (84056.18763327331, 62810.83680288951, 0.47217589093804146)
# 1-Feature Linear Test : (82431.00342204922, 61856.6306939777, 0.4786471230995255)

# ----------------- STEP7 Feature Scaling before using multiple features model -----------------
'''
    - Keep a note: scaling is **required** for kNN and your GD to behave well.
'''
from sklearn.discriminant_analysis import StandardScaler  
scaler = StandardScaler()
# fit only on training data, transform train & test , remember not to scale one hot encoded features
# alternative use ColumnTransformer 
# 1️⃣ Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=np.number).columns
categorical_features = X_train.select_dtypes(exclude=np.number).columns

# 2️⃣ Scale numeric features
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_train[numeric_features]),
    columns=numeric_features,
    index=X_train.index
)

X_test_num_scaled = pd.DataFrame(
    scaler.transform(X_test[numeric_features]),
    columns=numeric_features,
    index=X_test.index
)

# 3️⃣ Keep categorical features as-is
X_train_cat = X_train[categorical_features].copy()
X_test_cat = X_test[categorical_features].copy()

# 4️⃣ Concatenate scaled numeric + categorical
X_train = pd.concat([X_train_num_scaled, X_train_cat], axis=1)
X_test = pd.concat([X_test_num_scaled, X_test_cat], axis=1)

# Check numeric features scaling

# print("Train mean (after scaling):")
# print(X_train_scaled[numeric_features].mean()) # should be 0

# print("\nTrain std (after scaling):")
# print(X_train_scaled[numeric_features].std()) # should be 1

# ----------------- STEP8 Baseline 3: Multiple Linear Regression -----------------
# Train the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

lin_train_mulF = eval_metrics(y_train, y_train_pred)
lin_test_mulF = eval_metrics(y_test, y_test_pred)
# Train -> RMSE: 72287.42, MAE: 52700.69, R²: 0.610
# Test  -> RMSE: 70650.92, MAE: 52106.80, R²: 0.617

# ----------------- Step 9: Diagnostics -----------------
# # Diagnostics - Residuals
# residuals = y_test - y_test_pred

# # Residuals vs Predictions
# plt.figure(figsize=(6,4))
# sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.5)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel("Predicted values")
# plt.ylabel("Residuals (y - ŷ)")
# plt.title("Residuals vs Predicted")
# plt.show()

# # Histogram of residuals
# plt.figure(figsize=(6,4))
# sns.histplot(residuals, bins=30, kde=True)
# plt.xlabel("Residuals")
# plt.title("Distribution of Residuals")
# plt.show()

# # Coefficient interpretation
# coef_df = pd.DataFrame({
#     "Feature": X_train.columns,
#     "Coefficient": lin_reg.coef_
# }).sort_values(by="Coefficient", ascending=False)

# print("\nTop coefficients (standardized):")
# print(coef_df)

# ----------------- STEP10 Standardize (scale) target, Pick two features, Implement Batch GD Then inverse scale Target ----------------
# features = ["median_income", "latitude", "longitude", "bedrooms_per_household", "rooms_per_household"]
# features = X_train.columns.tolist()  # use all features
corrs = pd.DataFrame({
        "feature": X_train.columns,
        "corr": [np.corrcoef(X_train[f], y_train)[0,1] for f in X_train.columns]
    }).set_index("feature").abs().sort_values("corr", ascending=False)
features = corrs.index[:3]
x_train_scaled = X_train[features].copy()
y_train_scaled = y_train.copy()

y_scaled = scaler.fit_transform(y_train.values.reshape(-1,1)).flatten()

x_test_scaled = X_test[features].copy()
y_test_scaled = scaler.transform(y_test.values.reshape(-1,1)).flatten()

def gradient(x, y, theta):
    m = x.shape[0]
    return (2/m) * x.T.dot(x.dot(theta) - y)

def sgd(
    gradient, x, y, start=None, learn_rate=0.1,
    decay_rate=0.0, batch_size=1, n_iter=50, tolerance=1e-6,
    dtype="float64", random_state=None, return_cost=False
):
    if not callable(gradient):
        raise TypeError("Gradient must be callable")

    # Add intercept column
    x = np.c_[np.ones((x.shape[0], 1)), x]

    # Convert to correct dtype
    dtype_ = np.dtype(dtype)   
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    
    n_obs, n_vars = x.shape

    # Check parameters
    if learn_rate <= 0:
        raise ValueError("'learn_rate' must be > 0")
    if not (0 <= decay_rate <= 1):
        raise ValueError("'decay_rate' must be in [0,1]")
    if not 0 < batch_size <= n_obs:
        raise ValueError("'batch_size' must be in (0, n_obs]")
    if n_iter <= 0:
        raise ValueError("'n_iter' must be > 0")
    if tolerance <= 0:
        raise ValueError("'tolerance' must be > 0")

    # Random init
    rng = np.random.default_rng(seed=random_state)
    if start is None:
        vector = rng.normal(size=n_vars).astype(dtype_)
    else:
        vector = np.array(start, dtype=dtype_)

    # For momentum-like update
    diff = np.zeros_like(vector)

    # Track cost
    cost_history = []

    # Combine x and y for easy shuffling
    xy = np.c_[x, y.reshape(n_obs)]
    '''So y.reshape(n_obs, 1) forces y to be a column theta with n_obs rows and exactly one column.
    -1 is a special NumPy value that means “infer this size automatically.
    It works whether x started as:
    1D shape (n_obs,) → becomes (n_obs, 1)
    2D shape (n_obs, k) → stays (n_obs, k)
    1D shape (n_obs*k,) → becomes (n_obs, k) automatically
    ”'''
    stop_training = False
    for t in range(n_iter):
        rng.shuffle(xy)
        for start_idx in range(0, n_obs, batch_size):
            stop_idx = start_idx + batch_size
            x_batch, y_batch = xy[start_idx:stop_idx, :-1], xy[start_idx:stop_idx, -1]

            # Compute gradient using provided function
            grad = gradient(x_batch, y_batch, vector)

            # Momentum-like update
            diff = decay_rate * diff - learn_rate * grad
            
            # check convergence INSIDE batch loop
            if np.all(np.abs(diff) <= tolerance):
                stop_training = True
                break  
        #np.all(...) Returns True only if all parameters’ updates are smaller than the tolerance.
        # In other words, every parameter is barely changing anymore.
            # Update parameters
            vector += diff
        errors = x.dot(vector) - y
        cost = (1 / n_obs) * np.sum(errors**2)
        cost_history.append(cost)
        if stop_training:
            return (vector, cost_history) if return_cost else vector
    return (vector, cost_history) if return_cost else vector


vector, cost_history = sgd(
    gradient, x_train_scaled, y_scaled,
    start=None, learn_rate=0.05,
    decay_rate=1e-4, batch_size=256, n_iter=100, tolerance=1e-4,
    dtype="float64", random_state=42, return_cost=True
)


# ----------------- STEP11 Plot cost vs iterations ----------------
# '''You’ll see decreasing curve → convergence
# If learning rate is too high, cost may diverge → shows importance of lr'''
# batch_choices = [256, 512]
# learning_rates = [0.01, 0.05, 0.1]    # you can expand if needed
# decay = 1e-4
# n_iter = 100
# random_state = 42

# results = {}

# for b in batch_choices:
#     for lr in learning_rates:
#         key = f"b{b}_lr{lr}"
#         theta, cost_history = sgd(
#             gradient, x_train_scaled, y_scaled,
#             start=None, learn_rate=lr,
#             decay_rate=decay, batch_size=b,
#             n_iter=n_iter, tolerance=1e-6,
#             dtype="float64", random_state=random_state,
#             return_cost=True
#         )
#         results[key] = {"batch": b, "lr": lr, "cost": np.array(cost_history)}

# # Plot smoothed semilogy for clarity
# def moving_average(x, w=5):
#     if len(x) < w: return np.array(x)
#     return np.convolve(x, np.ones(w)/w, mode='valid')

# plt.figure(figsize=(12,6))
# for k, v in results.items():
#     ch = v["cost"]
#     smooth = moving_average(ch, w=5)
#     plt.semilogy(range(1, len(smooth)+1), smooth, label=f"{k}")
# plt.xlabel("Epoch")
# plt.ylabel("Cost (MSE, log scale)")
# plt.title("Batch size × Learning rate comparison")
# plt.legend(fontsize='small', ncol=2)
# plt.grid(alpha=0.3)
# plt.show()

# # Print summary
# print("Summary (final cost, epochs ran, updates per epoch):")
# for k, v in results.items():
#     b, lr = v["batch"], v["lr"]
#     ch = v["cost"]
#     print(f"{k:15} final={ch[-1]:.3e} epochs={len(ch)} updates_per_epoch={int(np.ceil(20640/b))}")
# we found out that b256_lr0.05     final=4.137e-01 epochs=100 updates_per_epoch=81 is the best choice

# ----------------- STEP12 Make predictions to compare with the test data ----------------
# Add intercept column to match vector length
X_train_bias = np.c_[np.ones((x_train_scaled.shape[0], 1)), x_train_scaled]
X_test_bias  = np.c_[np.ones((x_test_scaled.shape[0], 1)), x_test_scaled]

y_test_pred_sgd = X_test_bias.dot(vector)

# Inverse transform predictions back to original target scale:
y_test_pred_sgd = scaler.inverse_transform(y_test_pred_sgd.reshape(-1,1)).flatten()

test_metrics_sgd = eval_metrics(y_test, y_test_pred_sgd)

# print("SGD Test :", test_metrics_sgd)

# # ----------------- STEP12 Compare to the dummy regressors ----------------

# # -------------------
# # Feature selection experiment
# # -------------------
# def feature_selection_experiment(X_train, X_test, y_train, y_test):

#     results = {}

#     # --- 1. Correlation ranking ---
#     corrs = pd.DataFrame({
#         "feature": X_train.columns,
#         "corr": [np.corrcoef(X_train[f], y_train)[0,1] for f in X_train.columns]
#     }).set_index("feature").abs().sort_values("corr", ascending=False)

#     # --- 2. Best 1 feature ---
#     top1 = [corrs.index[0]]
#     model = LinearRegression()
#     model.fit(X_train[top1], y_train)
#     y_train_pred = model.predict(X_train[top1])
#     y_test_pred = model.predict(X_test[top1])
#     results["Top-1 Feature"] = {
#         "Train": eval_metrics(y_train, y_train_pred),
#         "Test": eval_metrics(y_test, y_test_pred),
#     }

#     # --- 3. Top 3 features ---
#     top3 = corrs.index[:3]
#     model = LinearRegression()
#     model.fit(X_train[top3], y_train)
#     y_train_pred = model.predict(X_train[top3])
#     y_test_pred = model.predict(X_test[top3])
#     results["Top-3 Features"] = {
#         "Train": eval_metrics(y_train, y_train_pred),
#         "Test": eval_metrics(y_test, y_test_pred),
#     }

#     # --- 4. All features ---
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_train_pred = model.predict(X_train)
#     y_test_pred = model.predict(X_test)
#     results["All Features"] = {
#         "Train": eval_metrics(y_train, y_train_pred),
#         "Test": eval_metrics(y_test, y_test_pred),
#     }

#     print("\nModel Evaluation Results (Feature Selection)\n")
#     print("{:<15} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
#         "Model", "Train RMSE", "Train MAE", "Train R²", "Test RMSE", "Test MAE", "Test R²"
#     ))
#     print("-" * 90)
#     for model_name, metrics in results.items():
#         train_rmse, train_mae, train_r2 = metrics["Train"]
#         test_rmse, test_mae, test_r2 = metrics["Test"]
#         print("{:<15} {:<12.2f} {:<12.2f} {:<12.4f} {:<12.2f} {:<12.2f} {:<12.4f}".format(
#             model_name, train_rmse, train_mae, train_r2, test_rmse, test_mae, test_r2
#         ))

#     return results, corrs

# feature_selection_experiment(X_train, X_test, y_train, y_test)
# # using only 3 features for sgd already changed the model

# '''Compare SGD train/test metrics vs:
# Dummy baseline
# Simple 1-feature regression
# Full LinearRegression (scikit-learn)'''
# results = {
#     "Dummy": {
#         "Train": dummy_train,
#         "Test": dummy_test
#     },
#     "1-Feature LR": {
#         "Train": lin_train_1f,
#         "Test": lin_test_1f
#     },
#     "Multi-Feature LR": {
#         "Train": lin_train_mulF,
#         "Test": lin_test_mulF
#     },
#     "SGD (Custom)": {
#         "Train": train_metrics_sgd,
#         "Test": test_metrics_sgd
#     }
# }

# print("\nModel Evaluation Results\n")
# header = "{:<20} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
#     "Model", "Train RMSE", "Train MAE", "Train R²",
#     "Test RMSE", "Test MAE", "Test R²"
# )
# print(header)
# print("-" * len(header))

# for model, scores in results.items():
#     tr_rmse, tr_mae, tr_r2 = scores["Train"]
#     te_rmse, te_mae, te_r2 = scores["Test"]
#     print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
#         model, tr_rmse, tr_mae, tr_r2, te_rmse, te_mae, te_r2
#     ))


# ============================================
# 9) kNN Regression (+ tuning)
# ============================================

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# -----------------------------
# 1. Pick features
# -----------------------------
# For clarity: let's start with a small subset
features_subset = ["median_income", "rooms_per_household"]  
X_train_knn = X_train[features_subset]
X_test_knn  = X_test[features_subset]

# -----------------------------
# 2. Define kNN model + GridSearch
# -----------------------------
# param_grid = {
#     "n_neighbors": list(range(50, 300)),   # try 50–300 neighbors
#     "weights": ["uniform", "distance"],  # two weighting schemes
#     "p": [1, 2]                          # 1=Manhattan, 2=Euclidean
# }

# knn = KNeighborsRegressor()

# grid_search = GridSearchCV(
#     knn,
#     param_grid,
#     cv=5,
#     scoring="neg_mean_squared_error",
#     n_jobs=-1, # This controls how many CPU cores are used while training/testing.
#     #verbose=1 # This controls how much info is printed while the search is running.
# )

# grid_search.fit(X_train_knn, y_train)


# -----------------------------
# 3. Best model and params : {'n_neighbors': 163, 'p': 2, 'weights': 'uniform'}
# # -----------------------------
# best_knn = grid_search.best_estimator_
# print("Best Params for Grid:", grid_search.best_params_)

# -----------------------------
# 4. Evaluate on test
# -----------------------------
knn = KNeighborsRegressor(n_neighbors=163, p=2, weights="uniform")
knn.fit(X_train_knn, y_train)
y_test_pred_knn  = knn.predict(X_test_knn)
print("KNN score for Grid: ", knn.score(X_test_knn, y_test))
print("KNN Test metrics for Grid: ", eval_metrics(y_test, y_test_pred_knn))

from scipy.stats import randint

param_grid_Rand = {
    "n_neighbors": randint(50, 300),   
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}

knn = RandomizedSearchCV(
    KNeighborsRegressor(),
    param_distributions=param_grid_Rand,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

knn.fit(X_train_knn, y_train)

best_CV = knn.best_estimator_
y_test_pred_Knn_RCV = best_CV.predict(X_test_knn)

print("Best Parameters:", knn.best_params_)
print("KNN score for Randomized Search: ", best_CV.score(X_test_knn, y_test))
print("KNN Test metrics for Randomized Search: ", eval_metrics(y_test, y_test_pred_Knn_RCV))

# -----------------------------
# 5. RMSE vs k plot (bias-variance tradeoff)
# -----------------------------
# mean_rmse = []
# mean_mae = []
# mean_r2 = []
# for k in range(150, 180):
#     knn_temp = KNeighborsRegressor(
#         n_neighbors=k,
#         weights="uniform",
#         p=2
#     )
#     knn_temp.fit(X_train_knn, y_train)
#     y_pred_temp = knn_temp.predict(X_test_knn)
#     rmse_temp, mae_temp, r2_temp = eval_metrics(y_test, y_pred_temp)
#     mean_rmse.append(rmse_temp)
#     mean_mae.append(mae_temp)
#     mean_r2.append(r2_temp)

# plt.figure(figsize=(8,5))
# plt.plot(range(150, 180), mean_rmse, marker="o")
# plt.xlabel("k (n_neighbors)")
# plt.ylabel("Test RMSE")
# plt.title("kNN Bias–Variance Tradeoff")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(range(150, 180), mean_mae, marker="o")
# plt.xlabel("k (n_neighbors)")
# plt.ylabel("Test mae")
# plt.title("kNN Bias–Variance Tradeoff")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,5))
# plt.plot(range(150, 180), mean_r2, marker="o")
# plt.xlabel("k (n_neighbors)")
# plt.ylabel("Test R^2")
# plt.title("kNN Bias–Variance Tradeoff")
# plt.grid(True)
# plt.show()

# # -----------------------------
# # 6. Optional: Visualization for 2D feature space
# # -----------------------------
# if X_train_knn.shape[1] == 2:
#     # Scatter plot with colormap predictions
#     plt.figure(figsize=(8,6))
#     scatter = plt.scatter(
#         X_test_knn.iloc[:,0],
#         X_test_knn.iloc[:,1],
#         c=y_test_pred_knn,
#         cmap="viridis",
#         s=35,
#         edgecolor="k"
#     )
#     plt.colorbar(scatter, label="Predicted Median House Value")
#     plt.xlabel(features_subset[0])
#     plt.ylabel(features_subset[1])
#     plt.title("kNN Predictions (color-mapped)")
#     plt.show()

# ------------------- Bagging --------------------------------
from sklearn.ensemble import BaggingRegressor
knn = KNeighborsRegressor(n_neighbors=163, p=2, weights="uniform")
bagging_knn = BaggingRegressor(estimator=knn, bootstrap=True, n_estimators=100, n_jobs=-1, random_state=42)
bagging_knn.fit(X_train_knn, y_train)
y_test_pred_bagging = bagging_knn.predict(X_test_knn)

print("KNN score for Bagging: ", bagging_knn.score(X_test_knn, y_test))
print("KNN Test metrics for Bagging: ", eval_metrics(y_test, y_test_pred_bagging))
'''KNN score for Grid:  0.5518548851642416
KNN Test metrics for Grid:  (np.float64(76424.75103078173), np.float64(56064.65690540733), 0.5518548851642416)
Best Parameters: {'n_neighbors': 168, 'p': 2, 'weights': 'uniform'}
KNN score for Randomized Search:  0.5522488558805401
KNN Test metrics for Randomized Search:  (np.float64(76391.15060409297), np.float64(56057.211804113605), 0.5522488558805401)
KNN score for Bagging:  0.5518128475552455
KNN Test metrics for Bagging:  (np.float64(76428.33540374221), np.float64(56071.579044294245), 0.5518128475552455)'''