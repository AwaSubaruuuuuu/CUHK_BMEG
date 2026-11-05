import pandas as pd
import numpy as np
# Loading data from train.csv file
train_df = pd.read_csv("train data credit card.csv")
train_df.head(5)

print(train_df.shape)


#train = train_df.loc[1:40000,:]
print(train_df.shape)


train = train_df.fillna("None")

train1 = train.drop(["ID","Is_Lead"],axis=1)
y = train["Is_Lead"]

print(train1.shape)
# import ordinal encoder from sklearn
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
  
# Transform the data
train1[["Gender","Region_Code","Occupation","Channel_Code","Credit_Product","Is_Active"]] = ord_enc.fit_transform(train1[["Gender","Region_Code","Occupation","Channel_Code","Credit_Product","Is_Active"]])

from sklearn.feature_selection import VarianceThreshold

var_thr = VarianceThreshold(threshold = 0.25) #Removing both constant and quasi-constant
var_thr.fit(train1)

print(var_thr.get_support())
#array([False,  True,  True,  True,  True,  True,  True,  True, False])

concol = [column for column in train1.columns 
          if column not in train1.columns[var_thr.get_support()]]

for features in concol:
    print(features)
    
    
train1=train1.drop(concol,axis=1)

print(train1.columns)
print(train1.shape)