import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('german.data-numeric', header = None, sep = '\s+')
df.iloc[:,-1][df.iloc[:,-1] ==2] = -1
dfr, dfs, yr, ys = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.33)
reg = LinearRegression().fit(dfr, yr)
reg_res = reg.predict(dfs)
reg_res[reg_res>0] = 1
reg_res[reg_res<0] = -1
log = LogisticRegression().fit(dfr, yr)
log_res = log.predict(dfs)
log_res[log_res>0] = 1
log_res[log_res<0] = -1
print('Linear Regression:')
print(classification_report(ys, reg_res, labels=[-1,1]))
print('Logistic Regression:')
print(classification_report(ys, log_res, labels=[-1,1]))
