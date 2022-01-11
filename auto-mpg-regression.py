import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import explained_variance_score

df = pd.read_csv("/home/nicholas/Repos/ML-Project/auto-mpg.tsv", sep = '\s+', names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'name'])

df = df[df.horsepower != "?"]

Y = df['mpg']

df_train, df_test, Y_train, Y_test = train_test_split(df, Y, test_size=0.25, random_state = 42)

features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']

features_year = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']

X_test = df_test[features]
X_train = df_train[features]

#model = svm.SVR()

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, Y_train)
predict = model.predict(X_test)

print("Accuracy: {0:.4f}".format(explained_variance_score(Y_test, predict)))