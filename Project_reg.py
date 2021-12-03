import pandas as pd
import seaborn
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif, mutual_info_classif, f_regression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, mean_squared_error
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
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
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
    ExtraTreesRegressor
)
from xgboost import XGBClassifier, XGBRFClassifier
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDRegressor

class Assignment:
    def __init__(self, data_file, profs_file=None, final_test=False):

        self.final_test = final_test
        self.data = self.load_data(data_file)
        self.features = self.data.iloc[:, 0:-2]
        self.target = self.data.iloc[:, -2] #take the kiba scores instead of the labels
        self.features_name = list(self.features)

        feature_select = 1
        if feature_select:
            d2 = pd.read_csv('./most_imp_feat_xgb.csv')
            imp = d2.iloc[:, 1].tolist()[:50]
            self.features = self.features[imp]
            self.features = MinMaxScaler().fit_transform(self.features)

        if self.final_test:
            print("take the professors data")
            self.test_data = self.load_data(profs_file)

    #data visualization
    def load_data(self, data):
        print("load the csv")
        try:
            print("load " + data)
            return pd.read_csv(data)
            #return pd.read_csv(data, header=None)
        except:
            print("ERROR: No file found")
            return None

    def log_csv(self, log):
        print("log the data")
        df = pd.DataFrame(data=log)
        print(df)
        df.index += 1
        print(df)
        df.to_csv('Group10_blind_predictions_egression.csv', index=True, header=False)

        #np.savetxt("Group10_blind_predictions.csv", log, delimiter=",")

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
        precision_inv = precision[::-1]
        recall_inv = recall[::-1]
        pr_res = np.interp(0.5, recall_inv, precision_inv)
        #pr_res = round(pr_res, 3)
        print("PR Score at recall of 50 is:" + str(pr_res))

        plt.figure()
        plt.axvline(0.5, 0, color="black", linestyle="dotted", label="Recall=0.5")
        plt.axhline(pr_res, 0, color="black", linestyle="dotted",
                    label=f"Precision={pr_res}")
        lw = 2
        plt.plot(recall, precision, color="darkorange", lw=lw, label="Precision Recall Curve", )
        plt.plot([0, 1], [0.5, 0.5], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        plt.legend(loc="lower right")

    def pred_from_kiba(self, pred):
        pred_cp = pred.copy()
        for i, kiba in enumerate(pred_cp):
            if kiba >= 12.1:
                pred_cp[i] = 1
            else:
                pred_cp[i] = 0

        return pred_cp


    def pred_from_kiba_df(self, pred):
        pred = pred.tolist()
        for i, kiba in enumerate(pred):
            if kiba >= 12.1:
                pred[i] = 1
            else:
                pred[i] = 0

        return pred

    def mse_calc(self, target, pred):
        mse = mean_squared_error(target, pred)
        print("mse = {}".format(mse))

class RegressionTree(Assignment):
    def __init__(self, data, fold=5, model=tree.DecisionTreeRegressor(max_depth=8), name='', regression=True, log=False):
        super().__init__(data)
        print(name)
        #self.features = self.feature_select()
        self.model = model
        #TODO add kfold test
        #self.target_prod = cross_val_predict(self.model, self.features, self.target, cv=fold)
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.074699)

        self.model.fit(X_train, y_train)

        self.target_pred = self.model.predict(X_test)
        #self.get_confusion_matrix_result(y_test, self.target_pred, "Decision Tree Classifier")

        #try:
        #self.target_prod = self.model.predict_proba(X_test)
        #print(self.target_prod[:,1])
        #self.roc_plot(y_test, self.target_prod[:,1])
        y_test_based_on_kiba = self.pred_from_kiba_df(y_test)
        self.target_pred_based_on_kiba = self.pred_from_kiba(self.target_pred)
        self.precision_recall_curve_plot(y_test_based_on_kiba, self.target_pred_based_on_kiba)
        # except:
        #    print("cannot get the probability or display the roc or prc")

        if self.final_test:
            print("Make prediction for the prof's data")
            self.target_pred = self.model.predict(self.test_data)

        self.mse_calc(y_test, self.target_pred)

        if log:
            self.log_csv(self.target_prod[:,1])
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

class Param(Assignment):
    def __init__(self,clf, data, param):
        super().__init__(data)
        self.model = RandomizedSearchCV(clf, param_distributions=param, cv=5)
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.5, train_size=0.33, stratify=self.target)
        self.model.fit(X_train, y_train)
        dt_opt = self.model.best_estimator_
        print("Parameter: ")
        print(dt_opt)

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
    kfold = 5

    '''models = [
        #("RF", RandomForestRegressor(max_depth=3, n_jobs=5)), PR Score at recall of 50 is:0.4155632466086149 mse = 0.6007702087856265
        #("GBDT", GradientBoostingRegressor(max_depth=3)), PR Score at recall of 50 is:0.5298694699790422 mse = 0.46706337150605187
        (
            "BUTIF",
            BaggingRegressor(
                BUTIF(
                    linear_model=LogisticRegression(max_iter=10000),
                    task="regression",
                    max_leaf=8,
                ),
                100,
                n_jobs=5,
            ),
        ),
        (
            "CO2",
            BaggingRegressor(
                ContinuouslyOptimizedObliqueRegressionTree(
                    MSE(), MeanSegmentor(), thau=500, max_iter=100, max_depth=3
                ),
                100,
                n_jobs=5,
            ),
        ),
        (
            "HHCART",
            BaggingRegressor(
                HouseHolderCART(MSE(), MeanSegmentor(), max_depth=3),
                100,
                n_jobs=5,
            ),
        ),
    ]

    for name, model in models:
        Classifier = RegressionTree(data, kfold, model, name=name, log=False)'''

    gb = GradientBoostingRegressor(max_depth=8, learning_rate=0.075, n_estimators=450)
    et = ExtraTreesRegressor(n_estimators=150, max_features="log2", min_samples_split=5)
    svm_sgd = SGDRegressor(max_iter=1000, tol=1e-3, loss='hinge')

    models = [('gb', gb), ('et', et)]
    vc = VotingRegressor(estimators=models)
    Classifier = RegressionTree(data, kfold, vc, log=False)

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