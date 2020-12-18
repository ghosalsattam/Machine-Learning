import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv('Position_Salaries.csv')
x=dataSet.iloc[:,1:2].values
y=dataSet.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)
yPred=regressor.predict([[6.5]])

xGrid=np.arange(min(x),max(x),.1)
xGrid=xGrid.reshape((len(xGrid),1))
plt.scatter(x,y,color="red")
plt.plot(xGrid,regressor.predict(xGrid))
