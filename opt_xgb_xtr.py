from google.colab import drive
drive.mount('/content/drive')
#%%
import pandas as pd
import numpy as np
#import seaborn as sns
import sklearn
import scipy.stats as sst
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
#%%
'''
data1 = pd.read_csv('/content/drive/My Drive/SYSC5405_Project/train_data.csv')
data2 = pd.read_csv('/content/drive/My Drive/SYSC5405_Project/imp_feat.csv')
'''
data1 = pd.read_csv(r'F:\UoOttawa Classes\Pattern Classification and Experimental Design\Project\train_data.csv')
data2 = pd.read_csv(r'F:\UoOttawa Classes\Pattern Classification and Experimental Design\Project\imp_feat.csv')
#%%
imp = data2.iloc[:,1].tolist()[:50]
print(imp)
#%%
X = data1[imp]
y = data1.Label
#%%
X_train, X_val, y_train, y_val=train_test_split(X, y, stratify=y, test_size=0.2)
#%%
# xgb=XGBClassifier(use_label_encoder=False, n_estimators=450, max_depth=20, learning_rate=0.075, subsample=1.0, gamma=0, colsample_bytree=0.1)
#%%
#xgb.fit(X_train,y_train)
#%%
xtr=ExtraTreesClassifier(n_estimators=150, max_features="log2",min_samples_split=5)
#%%
xtr.fit(X_train,y_train)
#%%
y_pred = xtr.predict(X_val)
#%%
precision, recall, thresho = precision_recall_curve(y_val, y_pred)
precision_inv=precision[::-1]
recall_inv=recall[::-1]
pr_res=np.interp(0.5,recall_inv,precision_inv)
pr_res=round(pr_res,3)
print("PR Score:" +str(pr_res))
#%%
'''
plot1=ConfusionMatrixDisplay.from_estimator(xgb,X_val,y_val)
plot2=PrecisionRecallDisplay.from_estimator(xgb,X_val,y_val,pos_label=1)
plot3=RocCurveDisplay.from_estimator(xgb,X_val,y_val,pos_label=1)
plt.show()
'''
#%%
plot1=ConfusionMatrixDisplay.from_estimator(xtr,X_val,y_val)
plot2=PrecisionRecallDisplay.from_estimator(xtr,X_val,y_val,pos_label=1)
plot3=RocCurveDisplay.from_estimator(xtr,X_val,y_val,pos_label=1)
plt.show()
#%%
#train_scores, valid_scores = validation_curve(estimator=xgb, X=X_val, y=y_val, param_name="subsample",param_range=subsample_range, scoring="precision")
#%%
train_scores, valid_scores = validation_curve(estimator=xtr, X=X_val, y=y_val, param_name="min_samples_split",param_range=min_samples_split_range, scoring="precision")
#%%
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)
diff_scores=train_scores_mean-valid_scores_mean
#%%
#n_estimators_range=[150,200,250]
#max_depth_range=[10,15,20,25,30]
#learning_rate_range=[0.07,0.0725,0.075,0.0775,0.08]
#subsample_range=[0.8,0.9,1.0]
#gamma_range=[0,1,5]
#colsample_bytree_range=[0.05,0.1]
#max_features_range=["sqrt", "log2"]
min_samples_split_range=[4,5,6,7]
#%%
'''
plt.plot(subsample_range, train_scores_mean, color='r', label='Training')
plt.plot(subsample_range, valid_scores_mean, color='b', label='Validation')
#plt.plot(subsample_range, diff_scores, color='g', label='Diff')
plt.xlabel("subsample")
plt.ylabel("precision")
plt.legend(loc="best")
plt.show()
'''
plt.plot(min_samples_split_range, train_scores_mean, color='r', label='Training')
plt.plot(min_samples_split_range, valid_scores_mean, color='b', label='Validation')
#plt.plot(min_samples_split_range, diff_scores, color='g', label='Diff')
plt.xlabel("min samples split")
plt.ylabel("precision")
plt.legend(loc="best")
plt.show()