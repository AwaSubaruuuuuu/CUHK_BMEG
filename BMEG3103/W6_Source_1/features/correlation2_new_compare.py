# correlation2_new.py - fixed version of correlation2.py
#
# Fix: the script imported load_boston but never used it, and
# load_boston was REMOVED in scikit-learn 1.2, so the import crashed.
# Just removing the dead import fixes the script - the rest reads train.csv.
#
# Added: compare model performance BEFORE vs AFTER dropping correlated features.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    classification_report,
)

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv("train.csv", nrows=10000)
X = df.drop(labels=["TARGET"], axis=1)
y = df["TARGET"]

# ----------------------------------------------------------------------
# 2. Train/test split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ----------------------------------------------------------------------
# 3. Correlation-based feature dropping
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


# Heatmap (optional, may be slow / huge if many features)
corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(11, 11)
sns.heatmap(corrmat)
plt.savefig("correlation2_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

corr_features = correlation(X_train, 0.9)
print(f"\nCorrelated features dropped ({len(corr_features)}): {sorted(corr_features)}")

# ----------------------------------------------------------------------
# 4. Keep ORIGINAL copy BEFORE dropping, for comparison
# ----------------------------------------------------------------------
X_train_orig = X_train.copy()
X_test_orig = X_test.copy()

n_features_before = X_train_orig.shape[1]

# Drop correlated features (same drop on train and test!)
X_train = X_train.drop(corr_features, axis=1)
X_test = X_test.drop(corr_features, axis=1)

n_features_after = X_train.shape[1]

print(f"\nFeature count: {n_features_before} -> {n_features_after}")
print("X_train shape after dropping:", X_train.shape)
print("X_test  shape after dropping:", X_test.shape)

# ----------------------------------------------------------------------
# 5. Compare model performance: BEFORE vs AFTER dropping
# ----------------------------------------------------------------------
def evaluate_classifier(model, X_tr, y_tr, X_te, y_te, name=""):
    """Fit model, predict, and return classification metrics."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # AUC needs probability scores
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, y_prob)
    else:
        auc = np.nan

    return {
        "model": name,
        "n_features": X_tr.shape[1],
        "AUC": auc,
        "Accuracy": accuracy_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
    }


results = []

# Logistic Regression
results.append(evaluate_classifier(
    LogisticRegression(max_iter=1000),
    X_train_orig, y_train, X_test_orig, y_test,
    name="LogReg - Original"
))
results.append(evaluate_classifier(
    LogisticRegression(max_iter=1000),
    X_train, y_train, X_test, y_test,
    name="LogReg - Reduced"
))

# Random Forest
results.append(evaluate_classifier(
    RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    X_train_orig, y_train, X_test_orig, y_test,
    name="RandomForest - Original"
))
results.append(evaluate_classifier(
    RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    X_train, y_train, X_test, y_test,
    name="RandomForest - Reduced"
))

df_results = pd.DataFrame(results)
print("\n===== Performance comparison: Original vs Reduced =====")
print(df_results.to_string(index=False))

# ----------------------------------------------------------------------
# 6. (Optional) Cross-validation for a more robust comparison
# ----------------------------------------------------------------------
def cv_evaluate(model, X, y, name="", cv=5, scoring="roc_auc"):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"{name}: {scoring} = {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores.mean()

print("\n===== 5-Fold Cross-Validation (AUC) =====")
cv_evaluate(LogisticRegression(max_iter=1000), X_train_orig, y_train, "LogReg - Original")
cv_evaluate(LogisticRegression(max_iter=1000), X_train, y_train, "LogReg - Reduced")
cv_evaluate(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
            X_train_orig, y_train, "RF - Original")
cv_evaluate(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
            X_train, y_train, "RF - Reduced")