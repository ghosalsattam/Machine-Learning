import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet1=pd.read_csv("LR_X.csv")
dataSet2=pd.read_csv("LR_Y.csv")
X=dataSet1.iloc[:,:]
Y=dataSet2.iloc[:,0]

from sklearn.linear_model import Ridge
for i in range(1,11):
    regressor=Ridge(alpha=i)
    regressor.fit(X,Y)
    y_pred=regressor.predict([[0,0,0],[1,0,0],[1,1,0],[1,1,1]])
    b0=y_pred[0]
    b1=y_pred[1]-y_pred[0]
    b2=y_pred[2]-b1-b0
    b3=y_pred[3]-b1-b2-b0
    plt.scatter(i,b1)
    plt.scatter(i,b2)
    plt.scatter(i,b3)