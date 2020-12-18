import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataSet=pd.read_csv("Position_Salaries.csv")

x=dataSet.iloc[:,1:2].values
y=dataSet.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

yPred=regressor.predict([[7.5]])

xGrid=np.arange(min(x),max(x),.01)
xGrid=xGrid.reshape((len(xGrid),1))
plt.scatter(x,y)
plt.plot(xGrid,regressor.predict(xGrid))