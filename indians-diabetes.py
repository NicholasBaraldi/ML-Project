import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/home/nicholas/Repos/ML-Project/diabetes.csv")
print(df.isnull().values.any())

df = df.append(pd.Series(), ignore_index=True)

if df.isnull().values.any():
    df = df.dropna()

df_train, df_test = train_test_split(df, test_size=0.25, shuffle=False)

df_test.drop("Outcome", inplace=True, axis=1)

y = df_train["Outcome"]

features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

X = pd.get_dummies(df_train[features])
X_test = pd.get_dummies(df_test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PersonID': df_test.index, 'Diabetes': predictions})
output.to_csv('submissionIndian.csv', index=False)
