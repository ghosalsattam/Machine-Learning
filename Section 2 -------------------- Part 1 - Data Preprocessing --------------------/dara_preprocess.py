#data pre processing
#import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import datas
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values