import numpy as np
import pandas as pd

df = pd.read_csv("/home/nicholas/Repos/ML-Project/diabetes.csv")
print(df.isnull().values.any())