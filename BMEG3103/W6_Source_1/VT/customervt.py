import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"Standard Customer Data.csv", nrows=40000)    #Taking only 40K rows from the start
print(data.shape)

train_x, test_x, train_y, test_y= train_test_split(data.drop("TARGET",axis=1),data.TARGET,test_size=0.2,random_state=41)

from sklearn.feature_selection import VarianceThreshold

constant_filter = VarianceThreshold(threshold=0)

#Fitand transforming on train data
data_constant = constant_filter.fit_transform(train_x)
print(data_constant.shape)

#Extracting all constant columns using get support function of our filter
constant_columns = [column for column in train_x.columns
                    if column not in train_x.columns[constant_filter.get_support()]]

#No. of constant columns
print(len(constant_columns))

#Constant columns names:
for column in constant_columns:
    print(column)
    
    
qcons_filter = VarianceThreshold(threshold=0.01)

#Fitand transforming on train data
data_qcons = qcons_filter.fit_transform(train_x)
print(data_qcons.shape)

#Extracting all Quasi constant columns using get support function of our filter
qcons_columns = [column for column in train_x.columns
                    if column not in train_x.columns[qcons_filter.get_support()]]

#No. of Quasi constant columns
print(len(qcons_columns))

#Quasi Constant columns names:
for column in qcons_columns:
    print(column)
    
data_qcons = data.drop(qcons_columns,axis=1)
print(data_qcons.shape)