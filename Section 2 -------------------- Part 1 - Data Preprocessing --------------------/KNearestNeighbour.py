import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataSet=pd.read_csv("Social_Network_Ads.csv")

x=dataSet.iloc[:,[2,3]].values
y=dataSet.iloc[:,4].values


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=.25,random_state=0)
xTrain=sc.fit_transform(xTrain)
xTest=sc.transform(xTest)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(xTrain,yTrain)

yPred=classifier.predict(xTest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)

from matplotlib.colors import ListedColormap
xSet,ySet=xTrain,yTrain
x1,x2=np.meshgrid(np.arange(start=xSet[:,0].min()-1,stop=xSet[:,0].max()+1,step=.01),
                  np.arange(start=xSet[:,1].min()-1,stop=xSet[:,1].max()+1,step=.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=.75,cmap=ListedColormap(('red','green')))
for i,j in enumerate(np.unique(ySet)):
    plt.scatter(xSet[ySet==j,0],xSet[ySet==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
xSet,ySet=xTest,yTest
x1,x2=np.meshgrid(np.arange(start=xSet[:,0].min()-1,stop=xSet[:,0].max()+1,step=.01),
                  np.arange(start=xSet[:,1].min()-1,stop=xSet[:,1].max()+1,step=.01))
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha=.75,cmap=ListedColormap(('red','green')))
for i,j in enumerate(np.unique(ySet)):
    plt.scatter(xSet[ySet==j,0],xSet[ySet==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
plt.legend()
plt.show()


