import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



data = pd.read_csv('training.csv', delimiter = ',')
data.head()
data.info()

#X= np.array(data.iloc[:,[1,2,3]])
X= data.iloc[:,[1,2,3]]
#y = np.array(data['Person'])
y = data['Person']
print(y.shape)


for i in range(0,y.shape[0]):
    if y[i] == "female":
        y[i] = 0
    else:
        y[i] = 1
        
print(X)
print(y)



y=y.astype('int')
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)

print(X_train)
print(y_train)


gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred))


x = np.array([[6, 130, 8]])
predicted= gnb.predict(x) # 0:female, 1:Male
print("Predicted Value:", predicted)


#calculation can be refered to 
#https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Sex_classification