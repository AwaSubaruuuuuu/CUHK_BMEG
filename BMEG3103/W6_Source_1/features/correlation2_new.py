# correlation2_new.py - fixed version of correlation2.py
#
# Fix: the script imported load_boston but never used it, and
# load_boston was REMOVED in scikit-learn 1.2, so the import crashed.
# Just removing the dead import fixes the script - the rest reads train.csv.

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv", nrows=10000)
X = df.drop(labels=["TARGET"], axis=1)
y = df["TARGET"]

from sklearn.model_selection import train_test_split

# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(labels=["TARGET"], axis=1),
    df["TARGET"],
    test_size=0.3,
    random_state=0,
)

print(X_train.shape)
print(X_test.shape)


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


import seaborn as sns

# Using Pearson Correlation
corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(11, 11)
sns.heatmap(corrmat)
plt.savefig("correlation2_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()

corr_features = correlation(X_train, 0.9)
print(f"Correlated features dropped ({len(corr_features)}): {corr_features}")

X_train = X_train.drop(corr_features, axis=1)
X_test = X_test.drop(corr_features, axis=1)

print(X_train.shape)
print(X_test.shape)

# original 371 columns
# (after dropping: see printed shapes above)
