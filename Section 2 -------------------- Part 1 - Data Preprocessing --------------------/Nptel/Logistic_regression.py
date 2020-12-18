import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet1=pd.read_csv("irisX.csv")
dataSet2=pd.read_csv("irisY.csv")
XTrain=dataSet1.iloc[0:99,:]
YTrain=dataSet2.iloc[0:99,:]
XTest=dataSet1.iloc[99:,:]
YTest=dataSet2.iloc[99:,0]
from sklearn.linear_model import LogisticRegression
for i in range(1,11):
    regressor=LogisticRegression(penalty="l2",C=4,multi_class="ovr")
    regressor.fit(XTrain,YTrain)
    
y_pred=regressor.predict(XTest)
count=0
#print(YTest[50])
for i in range(len(y_pred)):
    if(y_pred[i]==YTest[i+99]):
        count=count+1
p1=count/50
from sklearn.metrics import confusion_matrix,accuracy_score
p=accuracy_score(YTest,y_pred)

    