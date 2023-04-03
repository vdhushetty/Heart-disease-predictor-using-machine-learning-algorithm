import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from pandas import DataFrame, read_csv

heart_data = pd.read_csv("C:/Users/bhaga/Desktop/EEE 591 Python/Project 1/heart1.csv")
heart_data
corrMatrix = heart_data.corr().abs()
corrMatrix *= np.tri(*corrMatrix.values.shape, k=-1).T  # Reshaping the corr dataframe to not have repeated and diagonal values
corrMatrix_unstack = corrMatrix.unstack()  # Unstacking the corr dataframe
corrMatrix_unstack.sort_values(inplace=True, ascending=False)
print (corrMatrix_unstack.to_string())

covMatrix = heart_data.cov()
covMatrix = heart_data.cov().abs()
covMatrix *= np.tri(*covMatrix.values.shape, k=-1).T
print(covMatrix)
covMatrix_unstack = covMatrix.unstack()
covMatrix_unstack.sort_values(inplace=True,ascending=False)
print (covMatrix_unstack.to_string())

sn.set(style='whitegrid', context='notebook')  
sn.pairplot(heart_data, height=1.5, plot_kws={"s": 3})  
plt.show()