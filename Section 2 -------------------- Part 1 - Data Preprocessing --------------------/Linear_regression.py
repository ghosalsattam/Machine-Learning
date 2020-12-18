import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv('Salary_Data.csv')
print(dataSet)

x=dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,1].values

from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(x,y,test_size=.2,random_state=0)

#Single linear regression

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(XTrain,YTrain)

y_pred=regressor.predict(XTest)

plt.plot(XTest,y_pred,label="Test")
#plt.plot(XTrain,YTrain,label="Predicted")
plt.scatter(XTrain,YTrain,label="Predicted",color="red")
plt.show()

plt.scatter(XTest,YTest,color="red")
plt.plot(XTrain,regressor.predict(XTrain),color="green")