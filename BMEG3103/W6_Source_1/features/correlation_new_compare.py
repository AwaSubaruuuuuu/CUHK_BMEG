# correlation_new.py - fixed version of correlation.py
#
# Fix: sklearn.datasets.load_boston was REMOVED in scikit-learn 1.2
# (the Boston housing dataset was retired for ethical reasons).
# Replacement: fetch_california_housing (also a regression dataset).
#
# Added: compare model performance BEFORE vs AFTER dropping correlated features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# 1. Loading the dataset
# ----------------------------------------------------------------------
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target   # target: median house value

print("Feature names:", data.feature_names)
print(df.head())

X = df.drop("MedHouseVal", axis=1)   # Feature matrix
y = df["MedHouseVal"]

# ----------------------------------------------------------------------
# 2. Separate dataset into train and test
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

print("\nTrain/Test shapes:", X_train.shape, X_test.shape)
print("\nCorrelation matrix (train):")
print(X_train.corr())

# ----------------------------------------------------------------------
# 3. Visualize correlation heatmap
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 10))
cor = X_train.corr()
# NOTE: plt.cm.CMRmap_r is deprecated in newer matplotlib.
# Use a stable diverging colormap instead.
sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pearson Correlation Heatmap (Train Set)")
plt.savefig("correlation_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# 4. Find highly correlated features
# ----------------------------------------------------------------------
def correlation(dataset, threshold):
    col_corr = set()  # set of all the names of correlated columns
    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # we are interested in the absolute coefficient value
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # name of the column
                col_corr.add(colname)
    return col_corr


corr_features = correlation(X_train, 0.7)
print(f"\nCorrelated features dropped ({len(corr_features)}): {sorted(corr_features)}")

# ----------------------------------------------------------------------
# 5. Keep an ORIGINAL copy BEFORE dropping, for comparison
# ----------------------------------------------------------------------
X_train_orig = X_train.copy()
X_test_orig = X_test.copy()

print(f"\nOriginal number of features: {X_train_orig.shape[1]}")

# ----------------------------------------------------------------------
# 6. Drop correlated features (apply same drop to train and test!)
# ----------------------------------------------------------------------
X_train = X_train.drop(corr_features, axis=1)
X_test = X_test.drop(corr_features, axis=1)

print(f"Number of features after dropping: {X_train.shape[1]}")
print(f"Feature count: {X_train_orig.shape[1]} -> {X_train.shape[1]}")

print("\nX_train after dropping:")
print(X_train.head())
print("\nX_test after dropping:")
print(X_test.head())

# ----------------------------------------------------------------------
# 7. Compare model performance: BEFORE vs AFTER dropping
# ----------------------------------------------------------------------
def evaluate_model(model, X_tr, y_tr, X_te, y_te, name=""):
    """Fit model, predict, and return regression metrics."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return {
        "model": name,
        "n_features": X_tr.shape[1],
        "R2": r2_score(y_te, y_pred),
        "MAE": mean_absolute_error(y_te, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
    }


results = []

# Linear Regression
results.append(evaluate_model(
    LinearRegression(),
    X_train_orig, y_train, X_test_orig, y_test,
    name="LinearRegression - Original"
))
results.append(evaluate_model(
    LinearRegression(),
    X_train, y_train, X_test, y_test,
    name="LinearRegression - Reduced"
))

# Random Forest
results.append(evaluate_model(
    RandomForestRegressor(n_estimators=100, random_state=0),
    X_train_orig, y_train, X_test_orig, y_test,
    name="RandomForest - Original"
))
results.append(evaluate_model(
    RandomForestRegressor(n_estimators=100, random_state=0),
    X_train, y_train, X_test, y_test,
    name="RandomForest - Reduced"
))

df_results = pd.DataFrame(results)
print("\n===== Performance comparison: Original vs Reduced =====")
print(df_results.to_string(index=False))

# ----------------------------------------------------------------------
# 8. (Optional) Cross-validation for a more robust comparison
# ----------------------------------------------------------------------
from sklearn.model_selection import cross_val_score

def cv_evaluate(model, X, y, name="", cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"{name}: R2 = {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores.mean()

print("\n===== 5-Fold Cross-Validation (R2) =====")
cv_evaluate(LinearRegression(), X_train_orig, y_train, "Linear - Original")
cv_evaluate(LinearRegression(), X_train, y_train, "Linear - Reduced")
cv_evaluate(RandomForestRegressor(n_estimators=100, random_state=0),
            X_train_orig, y_train, "RF - Original")
cv_evaluate(RandomForestRegressor(n_estimators=100, random_state=0),
            X_train, y_train, "RF - Reduced")