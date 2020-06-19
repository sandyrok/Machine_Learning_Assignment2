import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('1D_regression_data.txt', header = None, sep = '\s+')
df = df.rename({0:1,1:6},axis = 1)
dfr, dfs, yr, ys = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.33)
l = [2,3,4,5,6,7,8,9,10]
x = 1
while True:
 reg = LinearRegression().fit(dfr, yr)
 reg_res = reg.predict(dfs)
 print(f'Degree {x}')
 print(f'Polynomial Coefficients:  {list(reg.coef_[::-1]) + [(reg.intercept_)]}')
 print("Test Error:")
 print('Mean Absolute Error:', metrics.mean_absolute_error(reg_res, ys))  
 print('Mean Squared Error:', metrics.mean_squared_error(reg_res, ys))  
 print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(reg_res, ys)))

 
 if l:
  x = l.pop(0)
  dfr[x] =  [ y ** x for y in dfr[1] ]
  dfs[x] =  [ y ** x for y in dfs[1] ]
  
 else:
  break

