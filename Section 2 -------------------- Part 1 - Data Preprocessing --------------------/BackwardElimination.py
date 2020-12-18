import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv('50_Startups.csv')

x=dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder=LabelEncoder()
x[:,3]=labelEncoder.fit_transform(x[:,3])

oneHotEncoder=OneHotEncoder(categorical_features=[3])

x=oneHotEncoder.fit_transform(x).toarray()

x=x[:,1:]
from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

Xtrn=XTrain[:,[2,4]]
regressor.fit(XTrain,YTrain)


yPred=regressor.predict(XTest)

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
xOpt=x[:,list(range(0,6))]
regressorOls=sm.OLS(endog=y,exog=xOpt).fit()
regressorOls.summary()

xOpt=x[:,[0,1,3,4,5]]
regressorOls=sm.OLS(endog=y,exog=xOpt).fit()
regressorOls.summary()

xOpt=x[:,[0,3,4,5]]
regressorOls=sm.OLS(endog=y,exog=xOpt).fit()
regressorOls.summary()

xOpt=x[:,[0,3,5]]
regressorOls=sm.OLS(endog=y,exog=xOpt).fit()
regressorOls.summary()

xOpt=x[:,[0,5]]
regressorOls=sm.OLS(endog=y,exog=xOpt).fit()
regressorOls.summary()