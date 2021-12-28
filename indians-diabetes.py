import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("/home/nicholas/Repos/ML-Project/diabetes.csv")
#print(df.isnull().values.any()) #validating if there is a null row

#df = df.append(pd.Series(), ignore_index=True) #adding a null row to test

#if df.isnull().values.any(): #droping null row
#    df = df.dropna()

Y = df["Outcome"].values

df_train, df_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.25, random_state = 42)

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

X_train = df_train[features]
X_test = df_test[features]

substitute_0_to_mean = SimpleImputer(missing_values = 0, strategy = "mean")

X_train = substitute_0_to_mean.fit_transform(X_train)
X_test = substitute_0_to_mean.fit_transform(X_test)

model_v1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model_v1.fit(X_train, Y_train)
predictions_v1 = model_v1.predict(X_test)

print("{0:.4f}".format(metrics.accuracy_score(Y_test, predictions_v1)))

gnb = GaussianNB() #Testing Naive Bayes model

model_v2 = gnb.fit(X_train, Y_train)
predictions_v2 = model_v2.predict(X_test)

print("{0:.4f}".format(metrics.accuracy_score(Y_test, predictions_v2)))

model_v3 = LogisticRegression(C = 0.7, max_iter = 140, random_state = 42) #Testing Regression model
model_v3.fit(X_train, Y_train)
predictions_v3 = model_v3.predict(X_test)

print("{0:.4f}".format(metrics.accuracy_score(Y_test, predictions_v3)))
