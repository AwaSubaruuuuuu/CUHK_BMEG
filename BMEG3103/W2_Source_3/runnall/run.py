import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
diabetes = pd.read_csv(script_dir / 'diabetes.csv')
print(diabetes.columns)

print("dimension of diabetes data: {}".format(diabetes.shape))



#KNN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'class'], diabetes['class'], stratify=diabetes['class'], random_state=66)
from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig(script_dir / 'knn_compare_model.png')

knn = KNeighborsClassifier(n_neighbors=9)#from graph
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))



#RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
print("Accuracy on RF training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on RF test set: {:.3f}".format(rf.score(X_test, y_test)))



#MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)
print("Accuracy on MLP training set: {:.2f}".format(mlp.score(X_train, y_train)))
print("Accuracy on MLP test set: {:.2f}".format(mlp.score(X_test, y_test)))


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp = MLPClassifier(
    hidden_layer_sizes=(20,),
    alpha=0.01,
    max_iter=2000,
    random_state=0,
)
mlp.fit(X_train_scaled, y_train)
print("Accuracy on MLP (Scale)training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on MLP(Scale) test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
