import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay


data = pd.read_csv(r'F:\UoOttawa Classes\Pattern Classification and Experimental Design\Project\train_data.csv')
x = data.drop('Label', axis=1)
y = data['Label']

pca = PCA()
clf = DecisionTreeClassifier()


pipe_hyp = Pipeline(steps=[("pca", pca), ("clf", clf)])


hyp_pam = {"pca__n_components": (5,10,15,20,25),
"clf__max_depth": (3,5,7,10,20,30,40,50,60),
"clf__criterion": ("gini","entropy"),
"clf__max_features": ("auto","sqrt","log2"),
"clf__min_samples_split": (2,4,6,8)
}

#DT_hyp = GridSearchCV(estimator=pipe_hyp, param_grid=hyp_pam, cv=5)
#DT_hyp=HalvingGridSearchCV(estimator=pipe_hyp, param_grid=hyp_pam, cv=5)
DT_hyp=RandomizedSearchCV(estimator=pipe_hyp, param_distributions=hyp_pam)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=1)
DT_hyp.fit(x_train, y_train)
dt_opt = DT_hyp.best_estimator_
print(dt_opt)