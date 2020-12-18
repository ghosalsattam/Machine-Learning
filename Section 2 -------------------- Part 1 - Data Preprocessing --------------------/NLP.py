import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#review=dataSet["Review"][3]
import re
from nltk.stem.porter import PorterStemmer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
corpus=[]
for i in range(1000):
    review=re.sub("[^a-zA-Z]"," ",dataSet["Review"][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    #stopwords=set(stopwords.words("english"))
    review=" ".join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(corpus).toarray()
y=dataSet.iloc[:,1].values

from sklearn.model_selection import train_test_split
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(xTrain,yTrain)

yPred=classifier.predict(xTest)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)