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
from sklearn.decomposition import PCA

dataset = pd.read_csv("C:\\Users\\bhard\\OneDrive\\Desktop\\SYSC 5405 Pattern Classification\\Project\\train_data.csv")



# print("Any NaN value in the dataset: ", dataset.isnull().values.any())
# print("Class", Counter(dataset.Label))
# print("Unique columns: ", len(set(dataset.columns)))

# # print(dataset.shape)    

# print("Duplicated rows: ", dataset.duplicated().sum())

## are there any categorical variables that need to be one hot encoded??
# dataset.describe()
# l=[]
# f=[]
# y = dataset.Label
# # jj = dataset.columns
# X = dataset.drop(["Label","KIBA"],axis="columns")
# for i in X.columns:
#     # print(i)
#     k = sst.spearmanr(X[i],dataset.KIBA)
#     if k[1]<0.05:
#         l.append(i)
#         f.append(k[0])
# # print(l)
# # print(len(l))
# print(sorted(f,reverse=True))

# X = X[l]
# print(X.shape)


# l=[]
# f=[]
# y = dataset.Label
# # jj = dataset.columns
# X = dataset.drop(["Label","KIBA"],axis="columns")
# j = 1
# kkk =[]
# for i in range(1,50,1):
#     kkk.append("G"+str(i))
# for i in X.columns:
#     # print(i)
#     # print("G"+str(j))
#     if i in kkk:
#         l.append(i)
#         # l.append(X[i])
#     j+=1

# print(l)
# print(len(l))
# # print(sorted(f,reverse=True))

# X = X[l]
# print(X.shape)


y = dataset.Label
# X = dataset.drop("Label",axis="columns")
X = dataset.drop(["Label","KIBA"],axis="columns")
# X = dataset.iloc[:,:13]
# X.head()




# a = np.sum(X.iloc[2,:],axis = 0)
# print(a/336)

## Applting PCA

# pca_X = PCA(n_components=200)
# X = pca_X.fit_transform(X)
# print(X.shape)
# X.head()

# X = SelectPercentile(X, percentile=30)

# sns.histplot(X.iloc[:,2])

# corr = X.corr()
# fig, ax = plt.subplots(figsize=(24, 18))
# sns.heatmap(corr, ax=ax)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state = 7, stratify=y)#random_state=2)#,shuffle=True)#, 


#%%
## AVENGERS "ENSEMBLE"

from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier


## encoding labels
# print(y_test[:10])

y_train.replace({False: 0, True: 1}, inplace=True)
y_test.replace({False: 0, True: 1}, inplace=True)

# print(y_test[:10])

ensemble_model = XGBClassifier(use_label_encoder=False)#, max_depth =7)
# ensemble_model = AdaBoostClassifier()#(max_depth = 7)
# ensemble_model = BaggingClassifier()
# ensemble_model = ExtraTreesClassifier()
# ensemble_model = GradientBoostingClassifier()
# ensemble_model = HistGradientBoostingClassifier()



# ensemble_modelrf = XGBRFClassifier()

ensemble_model.fit(X_train,y_train)
# ensemble_model.fit(X_train,y_train, eval_metric='aucpr')


# ensemble_modelrf.fit(X_train,y_train)

pred_y = ensemble_model.predict(X_test)
# pred_y = ensemble_modelrf.predict(X_test)

cm = confusion_matrix(y_test,pred_y)
tn,fp,fn,tp = cm.ravel()
# print(tn,fp,fn,tp)
# print("Before pruning:\n")
print("After pruning:\n")
print("Accuracy: ",(tp+tn)/(tp+tn+fp+fn))
print("Precision: ", tp/(tp+fp))
print("Recall: ", tp/(tp+fn))
print("Sensitivity: ", tp/(tp+fn))
print("Specificity: ",tn/(tn+fp))
pr = tp/(tp+fp)
rc = tp/(tp+fn)
print("F1 score: ", (2*pr*rc)/(pr+rc))


# ##ensemble_model
PrecisionRecallDisplay.from_estimator(ensemble_model,X_test,y_test,pos_label=1)
ConfusionMatrixDisplay.from_estimator(ensemble_model,X_test,y_test)
RocCurveDisplay.from_estimator(ensemble_model,X_test,y_test,pos_label=1)




#%%
#Feature selection


## Holdout testing (67/33)

# print(dataset.shape, X.shape, y.shape)

## 67% Training data, 33% test data

# print(X_train.shape)
# print(y_train.shape)

# print(X_test.shape)
# print(y_test.shape)

hyper_p = { "max_depth":(10,20,40,60),
"criterion":("gini","entropy"),
"max_features":("auto","sqrt","log2"),
"min_samples_split":(2,4,6,8),
"random_state": (3,5,7,9)
}


# hyper_p2 = { "max_depth":(10,20,40,60),
# "max_features":("auto","sqrt","log2"),
# } 

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

DT_all = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=hyper_p,  cv = 5, verbose=True)
# DT_all = GridSearchCV(DecisionTreeClassifier(), param_grid=hyper_p2,  cv = 5, verbose=True)

DT_all.fit(X_train,y_train)

DT_all.best_estimator_


#%%

# dt1 = DecisionTreeClassifier(max_depth=40, max_features="sqrt",min_samples_split=4,criterion="entropy",random_state=2)
 

# dt1 = DecisionTreeClassifier(class_weight=weights)#(max_depth=20, max_features="sqrt",min_samples_split=4,criterion="entropy")#,random_state=6)

# dt1 = DecisionTreeClassifier()#(max_depth=20, max_features="sqrt",min_samples_split=4,criterion="entropy")#,random_state=6)
# dt1 = DecisionTreeClassifier(max_depth=10, max_features="sqrt",min_samples_split=100,criterion="entropy",random_state=9)

# dt1 = DecisionTreeClassifier(max_depth=10, max_features="sqrt",min_samples_split=8,criterion="gini", class_weight=weights)
# dt1 = DecisionTreeClassifier()
# dt1 = DecisionTreeClassifier(max_depth=6, max_features="sqrt",min_samples_split=8,criterion="gini")#, class_weight=weights)

# weights = {0:8, 1:1}

dt1 = DecisionTreeClassifier(max_depth=7, min_samples_split=8, criterion="gini")#, class_weight="balanced")

dt1.fit(X_train,y_train)

pred_y = dt1.predict(X_test)

cm = confusion_matrix(y_test,pred_y)
tn,fp,fn,tp = cm.ravel()
# print(tn,fp,fn,tp)
# print("Before pruning:\n")
print("After pruning:\n")
print("Accuracy: ",(tp+tn)/(tp+tn+fp+fn))
print("Precision: ", tp/(tp+fp))
print("Recall: ", tp/(tp+fn))
print("Sensitivity: ", tp/(tp+fn))
print("Specificity: ",tn/(tn+fp))
pr = tp/(tp+fp)
rc = tp/(tp+fn)
print("F1 score: ", (2*pr*rc)/(pr+rc))

# pred_y_proba = dt1.predict_proba(X_test)[:,1]

# print(Counter((pred_y_proba)))

# # print(confusion_matrix(y_test,pred_y))

##performance on training data

# PrecisionRecallDisplay.from_estimator(dt1,X_train,y_train,pos_label=1)
# ConfusionMatrixDisplay.from_estimator(dt1,X_train,y_train)
# RocCurveDisplay.from_estimator(dt1,X_train,y_train,pos_label=1)

##performance on test data


PrecisionRecallDisplay.from_estimator(dt1,X_test,y_test,pos_label=1)
ConfusionMatrixDisplay.from_estimator(dt1,X_test,y_test)
RocCurveDisplay.from_estimator(dt1,X_test,y_test,pos_label=1)


# PrecisionRecallDisplay.from_predictions(y_test,pred_y_proba,pos_label=1)
# ConfusionMatrixDisplay.from_predictions(y_test,pred_y)

# print(average_precision_score(y_test, pred_y_proba,pos_label=1))



# ## need to check for overfitting

#%%
## post training pruning

pruning_path = dt1.cost_complexity_pruning_path(X_train,y_train)

alphas,impurities = pruning_path.ccp_alphas,pruning_path.impurities

_, ax2 = plt.subplots()

ax2.plot(alphas[:-1],impurities[:-1],marker = "x", drawstyle = "steps-post")


# print(len(alphas))

# dt_list = []

# for i in alphas:
#     dt_i = DecisionTreeClassifier(ccp_alpha = i,random_state = 0)
#     dt_list.append(dt_i.fit(X_train,y_train))


# train_scores = [dt.score(X_train, y_train) for dt in dt_list]
# test_scores = [dt.score(X_test, y_test) for dt in dt_list]
# fig, ax1 = plt.subplots()
# ax1.set_xlabel("Alpha")
# ax1.set_ylabel("Accuracy")
# ax1.set_title("Accuracy vs alpha for training and testing sets")
# ax1.plot(alphas, train_scores, marker='o', label="train",drawstyle="steps-post")
# ax1.plot(alphas, test_scores, marker='x', label="test",drawstyle="steps-post")
# ax1.legend()
# plt.show()

#%%
alp = 0.0009
alp = 0.001
alp2 = 0.0015

dt_new = DecisionTreeClassifier(ccp_alpha=alp2)
dt_new.fit(X_train,y_train)

PrecisionRecallDisplay.from_estimator(dt_new,X_test,y_test,pos_label=1)
ConfusionMatrixDisplay.from_estimator(dt_new,X_test,y_test)
RocCurveDisplay.from_estimator(dt_new,X_test,y_test,pos_label=1)


#%%
## Trying random forests

from sklearn.ensemble import RandomForestClassifier

rf1 = RandomForestClassifier(n_estimators=100, random_state=2)
rf1.fit(X_train,y_train)

PrecisionRecallDisplay.from_estimator(rf1,X_test,y_test,pos_label=1)
ConfusionMatrixDisplay.from_estimator(rf1,X_test,y_test)
RocCurveDisplay.from_estimator(rf1,X_test,y_test,pos_label=1)


#%%
## 5 fold cross validation

y = dataset.Label
# X = dataset.drop("Label",axis="columns")
X = dataset.drop(["Label","KIBA"],axis="columns")

## list to store k confusion matrices 
conf_mat_list = []

## Stratified K Fold
kfold_5 = StratifiedKFold(n_splits=5)
# pr, ax3 = plt.subplots(figsize=(7,5))
# roc, ax4 = plt.subplots(figsize=(7,5))

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

# print(Counter(pred_list))

# # precision, recall, _ = precision_recall_curve(real_list,pred_list,pos_label=1)
# # pl = PrecisionRecallDisplay(recall=recall, precision=precision, pos_label=1)  
# pl = PrecisionRecallDisplay.from_estimator(DecisionTreeClassifier,X,y,pos_label=1)
# a_p = np.round(average_precision_score(real_list,pred_list),2)
# pl.plot(ax=ax3,label="Average (final) (AP = "+str(a_p)+")",color="black")
# pr.suptitle("Precison-Recall curves")

# ff, tt, thresh = roc_curve(real_list,pred_list,pos_label=1)
# pl2 = RocCurveDisplay(fpr=ff, tpr=tt, pos_label=1)
# a_u_c = np.round(roc_auc_score(real_list,pred_list),2)
# pl2.plot(ax=ax4,label="Average (final) (AUC = "+str(a_u_c)+")",color="black")
# roc.suptitle("ROC curves ")
 

conf_mat_overall = np.sum(conf_mat_list, axis=0)
conf_mat_overall_disp = ConfusionMatrixDisplay(np.sum(conf_mat_list, axis=0))
conf_mat_overall_disp.plot(colorbar=False)
plt.title("DT classifier final confusion matrix (after 5 folds)")


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





