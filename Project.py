import pandas as pd
import seaborn
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif, mutual_info_classif, f_regression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn import tree
#import sklearn
import graphviz
import sys
import os
from dtreeviz.trees import dtreeviz
from scikit_obliquetree.BUTIF import BUTIF
from scikit_obliquetree.CO2 import ContinuouslyOptimizedObliqueRegressionTree
from scikit_obliquetree.GradientBoosting import GradientBoosting
from scikit_obliquetree.HHCART import HouseHolderCART
from scikit_obliquetree.segmentor import MSE, MeanSegmentor
from sklearn import model_selection
from sklearn.datasets import load_boston
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

class Assignment:
    def __init__(self, data_file):
        self.data = self.load_data(data_file)
        self.features = self.data.iloc[:, 0:-2]
        #self.features = self.data.iloc[:, 0:13]
        self.target = self.data.iloc[:, -1]
        self.features_name = list(self.features)

    #data visualization
    def load_data(self, data):
        try:
            return pd.read_csv(data)
            #return pd.read_csv(data, header=None)
        except:
            return None

    def plot_distribution(self, feature):
        #use seaborn
        feature = feature - 1
        plt.figure()
        seaborn.displot(data=self.data, x=feature, hue=(len(self.data.columns)-1))
        #plt.show()

    def plot_scatterplot(self, feature_a, feature_b):
        # use seaborn
        feature_a = feature_a -1
        feature_b = feature_b - 1
        plt.figure()
        print(self.data)
        seaborn.scatterplot(data=self.data, x=feature_a, y=feature_b, hue=(len(self.data.columns)-1))
        #plt.show()

    #preprocessing
    def select_k_best(self):
        data_new = SelectKBest(f_regression, k=20).fit_transform(self.features, self.target)
        return data_new

    def feature_select(self):
        selector = SelectFromModel(estimator=LogisticRegression()).fit(self.features, self.target)
        #print(selector.get_support())
        #print(str(selector.estimator_.coef_))
        #print("threshold: " + str(selector.threshold_))
        #selector.get_support()
        #Reduce X to the selected features.
        #selector.transform([self.features_name])
        #selector.transform(self.features)
        return selector.transform(self.features)

    def get_confusion_matrix_result(self, target, target_pred, title):
        cm = confusion_matrix(target, target_pred)
        TN, FP, FN, TP = cm.ravel()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        disp.ax_.set_title("Confusion Matrix {}".format(title))
        #accuracy
        ACC = (TP+TN) / (TP+TN+FN+FP)
        #precision
        PRE = TP / (TP+FP)
        #sensitivity
        Sn = TP / (TP+FN)
        #specificity
        Sp = TN / (TN+FP)
        print("Accuracy = {}, Precision = {}, Sensitivity = {}, Specificity = {}".format(ACC, PRE, Sn, Sp))

    def roc_plot(self, target, scores):
        fpr, tpr, thresholds = roc_curve(target, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc, )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")

    def precision_recall_curve_plot(self,target ,scores):
        precision, recall, thresholds = precision_recall_curve(target, scores, pos_label=1)
        plt.figure()
        lw = 2
        plt.plot(precision, recall, color="darkorange", lw=lw, label="Precision Recall Curve", )
        plt.plot([0, 1], [0.5, 0.5], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        plt.legend(loc="lower right")

    def pred_from_kiba(self, pred):
        for i, kiba in enumerate(pred):
            if pred > 12.1:
                pred[i] = 1
            else:
                pred[i] = 0


class DecisionTree(Assignment):
    def __init__(self, data, fold=5, model=tree.DecisionTreeRegressor(max_depth=8)):
        super().__init__(data)
        #self.features = self.select_k_best()
        self.model = model

        #self.target_prod = cross_val_predict(self.model, self.features, self.target, cv=fold)
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.5, train_size=0.1)
        #y_train.replace({'False': 0, 'True': 1}, inplace=True)
        #y_test.replace({'False': 0, 'True': 1}, inplace=True)
        self.model.fit(X_train, y_train)
        try:
            self.target_pred = self.model.predict(X_test)
            self.get_confusion_matrix_result(y_test, self.target_pred, "Decision Tree Classifier")
        except:
            print("cannot get the prediction or display the confusion matrix")
        try:
            self.target_prod = self.model.predict_proba(X_test)
            self.roc_plot(y_test, self.target_prod[:,1])
            self.precision_recall_curve_plot(y_test, self.target_prod[:,1])
        except:
            print("cannot get the probability or display the roc or prc")

        #PrecisionRecallDisplay.from_estimator(self.model, X_test, y_test, pos_label=1)
        #RocCurveDisplay.from_estimator(self.model, X_test, y_test, pos_label=1)
        #self.dt_visualisation()

    def dt_visualisation(self):
        viz = dtreeviz(self.model, self.features, self.target,
                       target_name="target",
                       feature_names=self.features_name,
                       show_node_labels=True
                       )
        viz.save("decision_tree.svg")

    def dt_graphivz(self):

        dot_data = tree.export_graphviz(self.model, out_file=None,
                                       feature_names=self.features_name,
                                       class_names=None,
                                       filled=True)

        # Draw graph
        graph = graphviz.Source(dot_data, format="png")
        graph.render("decision_tree_graphivz")

    def post_pruning(self, X_train, y_train):
        pruning_path = self.model.cost_complexity_pruning_path(X_train, y_train)
        alphas, impurities = pruning_path.ccp_alphas, pruning_path.impurities
        _, ax2 = plt.subplots()
        ax2.plot(alphas[:-1], impurities[:-1], marker="x", drawstyle="steps-post")



if __name__ == '__main__':
    print("Panda version = " + pd.__version__)
    print("Seaborn version = " + seaborn.__version__)
    #print("Graphvix version = " + graphviz.__version__)
    #print("sklearn version = " + sklearn.__version__)
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    #print(os.path.join())
    csv_file = 'train_data.csv'
    data_path = './Data/'

    data = data_path + csv_file
    #csv_file = 'assigData4.csv'

    #a4 = Assignment(data)
    #feature_to_compare = [1, 2]
    #a4.plot_scatterplot(feature_to_compare[0], feature_to_compare[1])


    kfold = 10
    Classifier = DecisionTree(data, kfold, BaggingClassifier(n_estimators=100, random_state=0))
    # RandomForestClassifier(max_depth=8)
    # GradientBoostingClassifier(max_depth=8)
    # tree.DecisionTreeClassifier(max_depth=3,min_samples_split=2, criterion="gini")
    # tree.DecisionTreeRegressor(max_depth=8)
    # BaggingClassifier(n_estimators=100, random_state=0)
    ''' BaggingRegressor(
            HouseHolderCART(MSE(), MeanSegmentor(), max_depth=12),
            n_estimators=100,
            n_jobs=5,
        )'''
    ''' BaggingRegressor(
        BUTIF(
            linear_model=LogisticRegression(max_iter=10000),
            task="regression",
            max_leaf=8,
        ),
        10,
        n_jobs=5,            
    )'''
    ''' BaggingRegressor(
        ContinuouslyOptimizedObliqueRegressionTree(
            MSE(), MeanSegmentor(), thau=500, max_iter=100, max_depth=8
        ),
        100,
        n_jobs=-1,
    )'''
    plt.show()


#issues
#https://stackoverflow.com/questions/62566691/error-when-loading-scipy-oserror-winerror-126-the-specified-module-could-not
#https://github.com/pytorch/ignite/issues/1153
#https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft

#download v2.49.0 (stable) from https://graphviz.org/download/
#add to path
#C:\Program Files\Graphviz
#install oblique tree
#https://pypi.org/project/scikit-obliquetree/