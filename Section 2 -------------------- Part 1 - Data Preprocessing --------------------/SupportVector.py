import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv('Position_Salaries.csv')
x=dataSet.iloc[:,1:2].values
y=dataSet.iloc[:,2].values

from sklearn.preprocessing import StandardScaler

scX=StandardScaler()
scY=StandardScaler()

x=scX.fit_transform(x)
y=scY.fit_transform(y.reshape(-1,1))


from sklearn.svm import SVR

regressor=SVR(kernel='rbf')
regressor.fit(x,y)


yPred=regressor.predict([[6.5]])
yPred=scY.inverse_transform(yPred)

plt.scatter(x,y,color="red")
plt.plot(x,regressor.predict(x),color="blue")