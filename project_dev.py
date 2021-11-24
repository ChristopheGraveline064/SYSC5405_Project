#%%


import pandas as pd
import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_selection import f_classif, chi2, SelectKBest, SelectPercentile
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve, roc_curve, RocCurveDisplay, roc_auc_score, average_precision_score


dataset = pd.read_csv("C:\\Users\\bhard\\OneDrive\\Desktop\\SYSC 5405 Pattern Classification\\Project\\train_data.csv")



print("Any NaN value in the dataset: ", dataset.isnull().values.any())
print("Class", Counter(dataset.Label))
print("Unique columns: ", len(set(dataset.columns)))

y = dataset.Label
X = dataset.drop("Label",axis="columns")

# sns.histplot(X.iloc[:,2])

# corr = X.corr()
# fig, ax = plt.subplots(figsize=(24, 18))
# sns.heatmap(corr, ax=ax)


#%%

## Holdout testing (67/33)

# print(dataset.shape, X.shape, y.shape)

## 67% Training data, 33% test data

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, stratify=y)

hyper_p = { "max_depth":(3,5,7,10,20,30,40,50,60),
"criterion":("gini","entropy"),
"max_features":("auto","sqrt","log2"),
"min_samples_split":(2,4,6,8)
}

from sklearn.model_selection import RandomizedSearchCV

DT_all = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=hyper_p,  cv = 5, verbose=True)

DT_all.fit(X_train,y_train)

DT_all.best_estimator_




#%%

dt1 = DecisionTreeClassifier(max_depth=40, max_features="sqrt",min_samples_split=8,criterion="entropy")
# dt1 = DecisionTreeClassifier(max_depth=1)

dt1.fit(X_train,y_train)

pred_y = dt1.predict(X_test)
pred_y_proba = dt1.predict_proba(X_test)[:,1]

# print(Counter((pred_y_proba)))

# # print(confusion_matrix(y_test,pred_y))


PrecisionRecallDisplay.from_estimator(dt1,X_test,y_test,pos_label=1)
ConfusionMatrixDisplay.from_estimator(dt1,X_test,y_test)
RocCurveDisplay.from_estimator(dt1,X_test,y_test,pos_label=1)


# PrecisionRecallDisplay.from_predictions(y_test,pred_y_proba,pos_label=1)
# ConfusionMatrixDisplay.from_predictions(y_test,pred_y)

# print(average_precision_score(y_test, pred_y_proba,pos_label=1))


# ## need to check for overfitting

#%%

## 5 fold cross validation


## list to store k confusion matrices 
conf_mat_list = []

## Stratified K Fold
kfold_5 = StratifiedKFold(n_splits=5)
pr, ax3 = plt.subplots(figsize=(7,5))
roc, ax4 = plt.subplots(figsize=(7,5))

real_list = []
pred_list = []

i=1
for train_i, test_i in kfold_5.split(X,y):

    X_train = X.iloc[train_i]
    y_train = y.iloc[train_i]
    
    X_test = X.iloc[test_i]
    y_test = y.iloc[test_i]
    
    # dt2 = DecisionTreeClassifier()
    # dt2 = DecisionTreeClassifier(max_depth=20)
    # dt2 = DecisionTreeClassifier(criterion="entropy")
    dt2 = DecisionTreeClassifier(class_weight="balanced")


    ## training

    dt2.fit(X_train.values,y_train)

    ## predicited labels

    predicted_y = dt2.predict(X_test.values)

    ## predicited probabilities

    predicted_y2 = dt2.predict_proba(X_test.values)
    
    real_list.append(y_test)
    pred_list.append(predicted_y2[:,1])
    conf_mat_list.append(confusion_matrix(y_test,predicted_y))

    i+=1

real_list = np.concatenate(real_list)
pred_list = np.concatenate(pred_list)
# print(pred_list)
# print(real_list.shape)

## complete tree will result in predicted probabilites as 0 and 1
## need to prune or limit depth

print(Counter(pred_list))

# precision, recall, _ = precision_recall_curve(real_list,pred_list,pos_label=1)
# pl = PrecisionRecallDisplay(recall=recall, precision=precision, pos_label=1)  
pl = PrecisionRecallDisplay.from_estimator(DecisionTreeClassifier,X,y,pos_label=1)
a_p = np.round(average_precision_score(real_list,pred_list),2)
pl.plot(ax=ax3,label="Average (final) (AP = "+str(a_p)+")",color="black")
pr.suptitle("Precison-Recall curves")

ff, tt, thresh = roc_curve(real_list,pred_list,pos_label=1)
pl2 = RocCurveDisplay(fpr=ff, tpr=tt, pos_label=1)
a_u_c = np.round(roc_auc_score(real_list,pred_list),2)
pl2.plot(ax=ax4,label="Average (final) (AUC = "+str(a_u_c)+")",color="black")
roc.suptitle("ROC curves ")
 

conf_mat_overall = np.sum(conf_mat_list, axis=0)
conf_mat_overall_disp = ConfusionMatrixDisplay(np.sum(conf_mat_list, axis=0))
conf_mat_overall_disp.plot(colorbar=False)
plt.title("DT classifier final confusion matrix (after 10 folds)")


tn,fp,fn,tp = conf_mat_overall.ravel()
# print(tn,fp,fn,tp)
print("Accuracy: ",(tp+tn)/(tp+tn+fp+fn))
print("Precision: ", tp/(tp+fp))
print("Recall: ", tp/(tp+fn))
print("Sensitivity: ", tp/(tp+fn))
print("Specificity: ",tn/(tn+fp))
pr = tp/(tp+fp)
rc = tp/(tp+fn)
print("F1 score: ", (2*pr*rc)/(pr+rc))


#%%

print(cross_val_score(DecisionTreeClassifier(), X, y, cv=3))





