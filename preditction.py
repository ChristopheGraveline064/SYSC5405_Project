#%%

import pandas as pd
import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt


# prec = pd.read_csv("C:\\Users\\bhard\\OneDrive\\Desktop\\SYSC5405_Project\\xgb,et,svm-10_pr@50rc.csv", names=["Precision"])

prec = pd.read_csv("C:\\Users\\bhard\\OneDrive\\Desktop\\SYSC5405_Project\\dt_pr@50rc.csv", names=["Precision"])



precision_values = prec["Precision"].tolist()

print(precision_values)
# print("For XGB- ET - SVM")

print("For DT Vanilla")

print("Mean: ",np.mean(precision_values))
print("STD: ",np.std(precision_values))
# print("VAR: ",np.var(precision_values))


