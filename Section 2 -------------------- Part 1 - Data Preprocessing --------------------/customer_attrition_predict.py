import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import optimizers
import numpy as np
import tensorflow as tf
data=pd.read_excel("Case Study Data.xls")
data['Weeks']=data['Weeks'].fillna('unknown')
data['Contract_Renewal']=data['Contract_Renewal'].fillna('unknown')
data['Data_Plan']=data['Data_Plan'].fillna('unknown')
data['Data_Usage']=data['Data_Usage'].fillna('unknown')
data['Calls_To_Customer_Care']=data['Calls_To_Customer_Care'].fillna('unknown')
data['DayMins']=data['DayMins'].fillna('unknown')
data['DayCalls']=data['DayCalls'].fillna('unknown')
data['MonthlyCharge']=data['MonthlyCharge'].fillna('unknown')
data['OverageFee']=data['OverageFee'].fillna('unknown')
data['RoamMins']=data['RoamMins'].fillna('unknown')
data['Customer_Attrition']=data['Customer_Attrition'].fillna('unknown')
data['Inserted_Date']=data['Inserted_Date'].fillna('unknown')
from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()

M=[]
X=[]
c=0

def fill_missing_values(data):
    data.iloc[:,12]=(labelEncoder.fit_transform(data.iloc[:,12]))
    for i in range(data.shape[0]):
        
        if(data.iloc[i,2]!='unknown'
           and data.iloc[i,3]!='unknown'
           and data.iloc[i,4]!='unknown'
           and data.iloc[i,5]!='unknown'
           and data.iloc[i,6]!='unknown'
           and data.iloc[i,7]!='unknown'
           and data.iloc[i,8]!='unknown'
           and data.iloc[i,9]!='unknown'
           and data.iloc[i,10]!='unknown'
           and data.iloc[i,11]!='unknown'
           and data.iloc[i,12]!='unknown'
           
           ):
            X.append(data.iloc[i,:])
        else:
            if(data.iloc[i,2]=='unknown' and data.iloc[i,3]=='unknown'):
                data.iloc[i,:13]=(data.iloc[i+1,:13]+data.iloc[i-1,:13])/2
            if(data.iloc[i,2]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[3,4,5,6,7,8,9,10,11,12]],data.iloc[:i,2])
                data.iloc[i,2]=(regressor.predict([data.iloc[i,[3,4,5,6,7,8,9,10,11,12]]]))[0]
            if(data.iloc[i,3]=='unknown'):
                classifier=RandomForestClassifier()
                regressor.fit(data.iloc[:i,[2,4,5,6,7,8,9,10,11,12]],data.iloc[:i,3])
                data.iloc[i,3]=(regressor.predict([data.iloc[i,[2,4,5,6,7,8,9,10,11,12]]]))
            if(data.iloc[i,4]=='unknown'):
                classifier=RandomForestClassifier()
                regressor.fit(data.iloc[:i,[2,3,5,6,7,8,9,10,11,12]],data.iloc[:i,4])
                data.iloc[i,4]=(regressor.predict([data.iloc[i,[2,3,5,6,7,8,9,10,11,12]]]))[0]
            if(data.iloc[i,5]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[2,3,4,6,7,8,9,10,11,12]],data.iloc[:i,5])
                data.iloc[i,5]=(regressor.predict([data.iloc[i,[2,3,4,6,7,8,9,10,11,12]]]))[0]
            if(data.iloc[i,6]=='unknown'):
                regressor=RandomForestClassifier()
                regressor.fit(data.iloc[:i,[2,3,4,5,7,8,9,10,11,12]],data.iloc[:i,6])
                data.iloc[i,6]=(regressor.predict([data.iloc[i,[2,3,4,5,7,8,9,10,11,12]]]))[0]
            if(data.iloc[i,7]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[2,3,4,5,6,8,9,10,11,12]],data.iloc[:i,7])
                data.iloc[i,7]=(regressor.predict([data.iloc[i,[2,3,4,5,6,8,9,10,11,12]]]))[0]
            if(data.iloc[i,8]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[2,3,4,5,6,7,9,10,11,12]],data.iloc[:i,8])
                data.iloc[i,8]=(regressor.predict([data.iloc[i,[2,3,4,5,6,7,9,10,11,12]]]))[0]
            if(data.iloc[i,9]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[2,3,4,5,6,7,8,10,11,12]],data.iloc[:i,9])
                data.iloc[i,9]=(regressor.predict([data.iloc[i,[2,3,4,5,6,7,8,10,11,12]]]))[0]
            if(data.iloc[i,10]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[2,3,4,5,6,7,8,9,11,12]],data.iloc[:i,10])
                data.iloc[i,10]=(regressor.predict([data.iloc[i,[2,3,4,5,6,7,8,9,11,12]]]))[0]
            if(data.iloc[i,11]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[2,3,4,5,6,7,8,9,10,12]],data.iloc[:i,11])
                data.iloc[i,11]=(regressor.predict([data.iloc[i,[2,3,4,5,6,7,8,9,10,12]]]))[0]
            if(data.iloc[i,12]=='unknown'):
                regressor=RandomForestRegressor()
                regressor.fit(data.iloc[:i,[2,3,4,5,6,7,8,9,10,11]],data.iloc[:i,12])
                data.iloc[i,12]=(regressor.predict([data.iloc[i,[2,3,4,5,6,7,8,9,10,11]]]))[0]
    return data    
data=fill_missing_values(data)
X=[]
'''
for i in range(data.shape[0]):
        
        if(data.iloc[i,2]!='unknown'
           and data.iloc[i,3]!='unknown'
           and data.iloc[i,4]!='unknown'
           and data.iloc[i,5]!='unknown'
           and data.iloc[i,6]!='unknown'
           and data.iloc[i,7]!='unknown'
           and data.iloc[i,8]!='unknown'
           and data.iloc[i,9]!='unknown'
           and data.iloc[i,10]!='unknown'
           and data.iloc[i,11]!='unknown'
           and data.iloc[i,12]!='unknown'
           
           ):
            X.append(data.iloc[i,:])
'''
x=data.iloc[:,[2,3,4,5,6,7,8,9,10,11]]
y=data.iloc[:,12]

XTrain,XTest,YTrain,YTest=train_test_split(x,y,test_size=.25,random_state=0)
XTrain=np.asarray(XTrain).astype('float32')
YTrain=np.asarray(YTrain).astype('float32')
XTest=np.asarray(XTest).astype('float32')
Ytest=np.asarray(YTest).astype('float32')
model=Sequential()
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
sgd=optimizers.SGD(lr=0.05, decay=1e-3, momentum=0.7, nesterov=True)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
epochs=200
history=model.fit(XTrain,YTrain,batch_size=150,epochs=epochs,validation_data=(XTest, YTest))
yPred=model.predict(XTest)
cm=confusion_matrix(YTest,yPred.round())
print("Confusion Matrix=",cm)
acc=history.history["accuracy"]
va=history.history["val_accuracy"]
plt.plot(range(1,epochs+1),acc,'r-')
plt.plot(range(1,epochs+1),va,'g-')
plt.legend(["Train","Test"])
plt.show()    
