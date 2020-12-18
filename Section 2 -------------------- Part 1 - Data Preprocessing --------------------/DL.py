import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv("Churn_Modelling.csv")
x=dataSet.iloc[:,3:13].values
y=dataSet.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoderx1=LabelEncoder()
x[:,1]=labelEncoderx1.fit_transform(x[:,1])
labelEncoderx2=LabelEncoder()
x[:,2]=labelEncoderx2.fit_transform(x[:,2])
oneHotEncoder=OneHotEncoder(categorical_features=[1])
x=oneHotEncoder.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xTrain=sc.fit_transform(xTrain)
xTest=sc.fit_transform(xTest)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()