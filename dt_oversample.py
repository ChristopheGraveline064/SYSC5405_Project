import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
from imblearn.over_sampling import RandomOverSampler


data = pd.read_csv(r'F:\UoOttawa Classes\Pattern Classification and Experimental Design\Project\train_data.csv')
x = data.drop('Label', axis=1)
y = data['Label']

x_train, x_test, y_train,  y_test = train_test_split(x, y, stratify=y, random_state=1)

clf = DecisionTreeClassifier(random_state=1)
ros = RandomOverSampler(sampling_strategy='minority')
x_resampled, y_resampled = ros.fit_resample(x, y)
clf.fit(x_resampled, y_resampled)
y_pred = clf.predict(x_test)
con_mat = confusion_matrix(y_test, y_pred)
disp_conmat = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=clf.classes_)
disp_pr = PrecisionRecallDisplay.from_predictions(y_test, y_pred)
disp_conmat.plot()
disp_pr.plot()
plt.show()

