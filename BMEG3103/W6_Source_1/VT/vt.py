import pandas as pd

ansur_male = pd.read_csv('ansur_male.csv', 
                         encoding='latin').drop('subjectid', axis=1)
#print(ansur_male.head())
print(ansur_male.shape)

from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold()

ansur_male_num = ansur_male.select_dtypes(include='number')  #only select numeric values

print(ansur_male_num.shape)

transformed = vt.fit_transform(ansur_male_num)
print(transformed.shape)


_ = vt.fit(ansur_male_num)
mask = vt.get_support()
ansur_male_num = ansur_male_num.loc[:, mask]


normalized_df = ansur_male_num / ansur_male_num.mean()
#print(normalized_df)

#print(normalized_df.var())

vt = VarianceThreshold(threshold=.003)

# Fit
_ = vt.fit(normalized_df)
print("D")
#print(_ )
#print("e")
# Get the mask
mask = vt.get_support()
print(mask)
# Subset the DataFrame
ansur_final = ansur_male_num.loc[:, mask]

print(ansur_final.shape)



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Build feature, target arrays
X, y = ansur_final.iloc[:, :-1], ansur_final.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1121218)

# Init, fit, score
forest = RandomForestRegressor(random_state=1121218)

_ = forest.fit(X_train, y_train)
print(X_train.shape)
# Training Score
print(f"Training Score: {forest.score(X_train, y_train)}")
print(f"Test Score: {forest.score(X_test, y_test)}")




# Build feature, target arrays
X, y = ansur_male_num.iloc[:, :-1], ansur_male_num.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1121218)

# Init, fit, score
forest = RandomForestRegressor(random_state=1121218)

_ = forest.fit(X_train, y_train)
print(X_train.shape)
# Training Score
print(f"Training Score: {forest.score(X_train, y_train)}")
print(f"Test Score: {forest.score(X_test, y_test)}")
