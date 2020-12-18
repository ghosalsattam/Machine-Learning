import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv('Position_Salaries.csv')
x=dataSet.iloc[:,1:2].values
y=dataSet.iloc[:,2].values

from sklearn.linear_model import LinearRegression
linReg=LinearRegression()
linReg.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
polyReg=PolynomialFeatures(degree=4)
xPoly=polyReg.fit_transform(x)
polyReg.fit(xPoly,y)

linReg2=LinearRegression()
linReg2.fit(xPoly,y)
plt.scatter(x,y,color="red")
plt.plot(x,linReg.predict(x),color="blue")
plt.scatter(x,y,color="red")
plt.plot(x,linReg2.predict(polyReg.fit_transform(x)),color="green")

linReg.predict([[6.5]])

linReg2.predict(polyReg.fit_transform([[6.5]]))
