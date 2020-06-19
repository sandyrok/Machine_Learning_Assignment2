import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import pandas as pd

#df = pd.read_csv('Gamma_train.txt', header = None)
#tst = pd.read_csv('Gamma_test.txt', header = None)

#df = pd.read_csv('Uniform_train.txt', header = None)
#tst = pd.read_csv('Uniform_test.txt', header = None)

df = pd.read_csv('Normal_train_10D.txt', header = None)
tst = pd.read_csv('Normal_test_10D.txt', header = None)

for i in [10,50,100,500]:
 sp = df.sample(i)
 reg = LinearRegression().fit(sp.iloc[:,0:-1], sp.iloc[:,-1])
 reg_res = reg.predict(tst.iloc[:,0:-1])
 reg_res[reg_res>0] = 1
 reg_res[reg_res<0] = -1
 log = LogisticRegression().fit(sp.iloc[:,0:-1], sp.iloc[:,-1])
 log_res = log.predict(tst.iloc[:,0:-1])
 log_res[log_res>0] = 1
 log_res[log_res<0] = -1
 print('Linear Regression:')
 print(classification_report(tst.iloc[:,-1], reg_res, labels=[-1,1]))
 print('Logistic Regression:')
 print(classification_report(tst.iloc[:,-1], log_res, labels=[-1,1]))
