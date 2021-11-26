import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
from imblearn.over_sampling import RandomOverSampler


data = pd.read_csv(r'F:\UoOttawa Classes\Pattern Classification and Experimental Design\Project\train_data.csv')
x = data.drop(labels=['KIBA','Label'], axis=1)
y = data['Label']

#x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)
#x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size=0.25, random_state=1)
#val=0.2, train=0.6, test=0.2
x_train, x_val, y_train, y_val=train_test_split(x, y, test_size=0.2, stratify=y)

clf = DecisionTreeClassifier()
#ros = RandomOverSampler(sampling_strategy='minority')

depth_range=(2,3,4,5,6,7,8,9,10)
#min_samples_range=[5,6,7,8,9,10]
train_scores, valid_scores = validation_curve(estimator=clf, X=x_val, y=y_val, param_name="max_depth",param_range=depth_range)
#train_scores, valid_scores = validation_curve(estimator=clf, X=x_val, y=y_val, param_name="min_samples_split",param_range=min_samples_range)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.plot(depth_range, train_scores_mean, color='r', label='Training')
plt.fill_between(depth_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="darkorange")
plt.plot(depth_range, valid_scores_mean, color='b', label='Validation')
plt.fill_between(depth_range, valid_scores_mean - valid_scores_std, train_scores_mean + valid_scores_std, color="green")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()
'''
plt.plot(min_samples_range, train_scores_mean, color='r', label='Training')
#plt.fill_between(min_samples_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color="darkorange")
plt.plot(min_samples_range, valid_scores_mean, color='b', label='Validation')
#plt.fill_between(min_samples_range, valid_scores_mean - valid_scores_std, train_scores_mean + valid_scores_std, color="green")
plt.xlabel("Min Samples Split")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.show()
'''
'''
x_resampled, y_resampled = ros.fit_resample(x_train, y_train)
clf.fit(x_resampled, y_resampled)
'''
#clf.fit(x_train, y_train)
'''
y_pred = clf.predict(x_test)

con_mat = confusion_matrix(y_test, y_pred)
disp_conmat = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=clf.classes_)
disp_pr = PrecisionRecallDisplay.from_predictions(y_test, y_pred)
disp_conmat.plot()
disp_pr.plot()
plt.show()
'''






