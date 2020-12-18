import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet=pd.read_csv("Salary_Data.csv")

ar=np.array(dataSet)
me=np.mean(ar[:,1])

mi=np.min(ar[:,1])
ma=np.max(ar[:,1])

