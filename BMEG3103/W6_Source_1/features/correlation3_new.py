# correlation3_new.py - fixed version of correlation3.py
#
# Fixes:
# 1. pandas: assigning into a slice triggers SettingWithCopyWarning /
#    silently fails under copy-on-write -> use .copy() and assign safely
# 2. sklearn: SVC.fit() needs a 1-D label vector -> .ravel()
# 3. BUG in the original: the second section encoded data.iloc[:,0]
#    (the FIRST dataframe) instead of data1.iloc[:,0]

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

np.random.seed(123)

data = pd.read_csv("cancer.csv")

data = data.iloc[:, 1:-1].copy()   # .copy(): avoid pandas slice-assignment issues

label_encoder = LabelEncoder()
# encode the diagnosis column (M/B -> 1/0) safely
data[data.columns[0]] = label_encoder.fit_transform(data[data.columns[0]]).astype("float64")

print(data.shape)

corr = data.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if corr.iloc[i, j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = data.columns[columns]

selected_columns = selected_columns[1:].values

data = data.drop(selected_columns, axis=1)
print(data.shape)

result = pd.DataFrame()
result["diagnosis"] = data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(
    data.iloc[:, 1:], result["diagnosis"].values, test_size=0.2   # 1-D labels
)
print(x_train.shape)
print(x_test.shape)

svc = SVC()  # the default kernel used by SVC is the gaussian kernel
svc.fit(x_train, y_train)

prediction = svc.predict(x_test)

cm = confusion_matrix(y_test, prediction)
correct = 0
for i in range(cm.shape[0]):
    correct += cm[i][i]

accuracy = correct / x_test.shape[0]
print("Accuracy (correlation-selected features):", accuracy)


##############################################################################################
# Second run: use ALL features (no correlation-based selection) for comparison

data1 = pd.read_csv("cancer.csv")
result = pd.DataFrame()
result["diagnosis"] = data1.iloc[:, 1]
data1 = data1.iloc[:, 2:-1].copy()

label_encoder = LabelEncoder()
# FIX: encode data1 (was wrongly 'data' in the original script)
data1[data1.columns[0]] = label_encoder.fit_transform(
    data1[data1.columns[0]]).astype("float64")

x_train, x_test, y_train, y_test = train_test_split(
    data1.values, result["diagnosis"].values, test_size=0.2
)
svc = SVC()
svc.fit(x_train, y_train)
print(x_train.shape)

prediction = svc.predict(x_test)
cm = confusion_matrix(y_test, prediction)
correct = 0
for i in range(cm.shape[0]):
    correct += cm[i][i]

accuracy = correct / x_test.shape[0]
print("Accuracy (all features):", accuracy)
