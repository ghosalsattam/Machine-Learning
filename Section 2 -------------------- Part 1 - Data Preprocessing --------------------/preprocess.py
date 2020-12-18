import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#set working directory
#create dataset

dataset = pd.read_csv('Data.csv')
#seprating Independent Veriable
x=dataset.iloc[:,:-1].values
#sepratingdependent Veriable
y=dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#taking care od Categrical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_x = LabelEncoder()
x[:,0]=labelEncoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
labelEncoder_y = LabelEncoder()
y=labelEncoder_y.fit_transform(y)

#spliting data into Training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

