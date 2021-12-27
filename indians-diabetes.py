import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("/home/nicholas/Repos/ML-Project/diabetes.csv")
#print(df.isnull().values.any()) #validating if there is a null row

#df = df.append(pd.Series(), ignore_index=True) #adding a null row to test

#if df.isnull().values.any(): #droping null row
#    df = df.dropna()

Y = df["Outcome"].values

df_train, df_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.25, random_state = 42)

df_test.drop("Outcome", inplace=True, axis=1)
df_train.drop("Outcome", inplace=True, axis=1)

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

X_train = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

output = pd.DataFrame({'PersonID': df_test.index, 'Diabetes': predictions})
output.to_csv('submissionIndian.csv', index=False)

print("{0:.4f}".format(metrics.accuracy_score(Y_test, predictions)))


gnb = GaussianNB() #Testing Naive Bayes model

model = gnb.fit(df_train, Y_train)
predictions = model.predict(df_test)

print("{0:.4f}".format(metrics.accuracy_score(Y_test, predictions)))