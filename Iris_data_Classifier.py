import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv('iris_dataset.txt', header = None)
dfo  = df.copy()
dfo[4] = df[4].apply(lambda x: [ 1 if i == x else 0 for i in range(1,4)])
print(dfo.head(100))
