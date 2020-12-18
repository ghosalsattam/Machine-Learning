import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv('50_Startups.csv')

x=dataSet.iloc[:,:-1].values
y=dataSet.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder=LabelEncoder()
x[:,3]=labelEncoder.fit_transform(x[:,3])

onehotencoder=OneHotEncoder(categorical_features=[3])

x=onehotencoder.fit_transform(x).toarray()

x=x[:,1:]
from sklearn.model_selection import train_test_split
XTrain,XTest,YTrain,YTest=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(XTrain,YTrain)


yPred=regressor.predict(XTest)

