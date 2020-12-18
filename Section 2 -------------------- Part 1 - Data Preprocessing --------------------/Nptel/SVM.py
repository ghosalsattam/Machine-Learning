import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataSet1=pd.read_csv("irisX.csv",header=None)
dataSet2=pd.read_csv("irisY.csv",header=None)
XTrain=dataSet1.iloc[0:100,:]
YTrain=dataSet2.iloc[0:100,:]
XTest=dataSet1.iloc[100:,:]
YTest=dataSet2.iloc[100:,0]
#print(XTrain[0])
from sklearn.svm import SVC
regressor=SVC(C=1,gamma=.5,kernel='rbf')
regressor.fit(XTrain,YTrain)
y_pred=regressor.predict(XTest)
count=0
#a=regressor.n_support_
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(YTest,y_pred)

p=accuracy_score(YTest,y_pred)