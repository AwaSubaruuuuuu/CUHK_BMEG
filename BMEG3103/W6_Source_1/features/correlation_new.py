# correlation_new.py - fixed version of correlation.py
#
# Fix: sklearn.datasets.load_boston was REMOVED in scikit-learn 1.2
# (the Boston housing dataset was retired for ethical reasons).
# Replacement: fetch_california_housing (also a regression dataset).

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Loading the dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target   # target: median house value

print(data.feature_names)
print(df.head())

X = df.drop("MedHouseVal", axis=1)   # Feature matrix
y = df["MedHouseVal"]

# separate dataset into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

print(X_train.shape, X_test.shape)
print(X_train.corr())

import seaborn as sns

# Using Pearson Correlation
plt.figure(figsize=(12, 10))
cor = X_train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.CMRmap_r)
plt.savefig("correlation_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()


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
print(f"Correlated features dropped ({len(corr_features)}): {corr_features}")

print(f"Original number of features: {X_train.shape[1]}")
X_train = X_train.drop(corr_features, axis=1)
X_test = X_test.drop(corr_features, axis=1)

print(X_train)
print(X_test)



print(f"Number of features after dropping: {X_train.shape[1]}")
