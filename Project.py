import pandas as pd
import seaborn
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_classif, mutual_info_classif, f_regression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
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
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    ExtraTreesClassifier
)
from xgboost import XGBClassifier, XGBRFClassifier
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier

class Assignment:
    def __init__(self, data_file, profs_file=None, final_test=False):
        self.final_test = final_test
        self.data = self.load_data(data_file)
        self.features = self.data.iloc[:, 0:-2]
        self.target = self.data.iloc[:, -1]
        self.features_name = list(self.features)

        feature_select = 1
        if feature_select:
            d2 = pd.read_csv('./most_imp_feat_xgb.csv')
            imp = d2.iloc[:, 1].tolist()[:100]
            self.features = self.features[imp]
            self.features = MinMaxScaler().fit_transform(self.features)

        if self.final_test:
            print("take the professors data")
            self.test_data = self.load_data(profs_file)

    #data visualization
    def load_data(self, data):
        print("load the csv")
        try:
            return pd.read_csv(data)
            #return pd.read_csv(data, header=None)
        except:
            return None

    def log_csv(self, log, Name='Group10_blind_predictions.csv'):
        print("log the data " + Name)
        df = pd.DataFrame(data=log)
        #print(df)
        df.index += 1
        #print(df)
        df.to_csv(Name, index=True, header=False)

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
        '''plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc, )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")'''

    def precision_recall_curve_plot(self,target ,scores):
        precision, recall, thresholds = precision_recall_curve(target, scores, pos_label=1)
        precision_inv = precision[::-1]
        recall_inv = recall[::-1]
        pr_res = np.interp(0.5, recall_inv, precision_inv)
        #pr_res = round(pr_res, 3)
        print("PR Score at recall of 50 is:" + str(pr_res))

        '''plt.figure()
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
        plt.legend(loc="lower right")'''
        return pr_res

    def pred_from_kiba(self, pred):
        for i, kiba in enumerate(pred):
            if pred > 12.1:
                pred[i] = 1
            else:
                pred[i] = 0


class DecisionTree(Assignment):
    def __init__(self, data, fold=5, model=tree.DecisionTreeClassifier(max_depth=8), eval='', regression=False, log=False):
        super().__init__(data)
        #self.features = self.feature_select()
        self.model = model
        #TODO add kfold test
        #self.target_prod = cross_val_predict(self.model, self.features, self.target, cv=fold)
        if regression:
            #TODO add a regression method and a metric to mesure its performance
            pass
        else:
            #X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.5, train_size=0.33, stratify=self.target)
            X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size = 0.074699, stratify = self.target)

        self.model.fit(X_train, y_train)
        '''try:
            print("try to fit with aucpr")
            self.model.fit(X_train, y_train, eval_metric='aucpr')
        except:
            self.model.fit(X_train, y_train)'''
        try:
            self.target_pred = self.model.predict(X_test)
            self.get_confusion_matrix_result(y_test, self.target_pred, "Decision Tree Classifier")
        except:
            print("cannot get the prediction or display the confusion matrix")
        #try:
        self.target_prod = self.model.predict_proba(X_test)
        print(self.target_prod[:,1])
        self.roc_plot(y_test, self.target_prod[:,1])
        self.p_at_r_50 = self.precision_recall_curve_plot(y_test, self.target_prod[:,1])
        # except:
        #    print("cannot get the probability or display the roc or prc")

        if self.final_test:
            print("Make prediction for the prof's data")
            self.target_prod_prof = self.model.predict_proba(self.test_data)

        if log:
            self.log_csv(self.target_prod[:, 1], Name='Group10_blind_predictions.csv')
            if self.final_test:
                self.log_csv(self.target_prod_prof[:, 1], Name='Group10_blind_predictions_regression.csv')

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

    DT = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=6, min_samples_leaf=4, max_depth=7, max_features="sqrt")
    #tree = tree.DecisionTreeRegressor(max_depth=8)
    '''clf = MLPClassifier(activation='tanh', hidden_layer_sizes=170, learning_rate=0.003)
    hyp_pam = {
        "learning_rate_init": (0.003),
    }
    Classifier = Param(clf, data, hyp_pam)'''

    '''clf = ExtraTreesClassifier(  n_estimators=200)
    hyp_pam = {
               #"n_estimators": (10, 20, 50, 100, 200),
               "criterion": ("gini", "entropy"),
               "max_features": ("auto", "sqrt", "log2"),
               "min_samples_split": (2, 4, 6, 8),
               "max_depth": (60, 100 , 150, 200),
               }

    Classifier = Param(clf, data, hyp_pam)'''
    xgb = XGBClassifier(use_label_encoder=False, max_depth=20, learning_rate=0.075, n_estimators=450, scale_pos_weight=1.5)#, random_state =42)  # scale_pos_weight = 86324/23155)#, eval_metric = "error" #"logloss")#, max_depth =7)
    lr = LogisticRegression(max_iter=1000, solver='saga')  ##can update max_iter, solver = saga
    ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    sv = SVC(probability=True)
    sv2 = LinearSVC()
    et = ExtraTreesClassifier(n_estimators=150, max_features="log2", min_samples_split=5)#, random_state=42)
    dt = tree.DecisionTreeClassifier(max_depth=7)
    knn = KNeighborsClassifier(n_neighbors=9)
    bg = BaggingClassifier(DT, n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1)
    rf = RandomForestClassifier(max_depth=6, n_estimators=500, max_leaf_nodes=10, n_jobs=-1)
    mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=170, learning_rate_init=0.003)
    svm_sgd = CalibratedClassifierCV(SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge'))
    lr_sgd = CalibratedClassifierCV(SGDClassifier(max_iter=1000, tol=1e-3, loss='log', random_state=42))

    #xgb = XGBClassifier(use_label_encoder=False, max_depth=6, learning_rate=0.3, n_estimators=500, scale_pos_weight=1.5)
    #et = ExtraTreesClassifier(max_depth=6)
    #bag = BaggingClassifier(tree, n_estimators=100, bootstrap=False, bootstrap_features=True)
    #sv = SVC(probability=True, kernel="linear")
    #xgb = XGBClassifier(use_label_encoder=False, n_estimators=450, max_depth=20, learning_rate=0.075, subsample=1.0, gamma=0, colsample_bytree=0.1)

    #models = [('xgb', xgb), ('et', et), ('sgd', svm_sgd), ("lr_sgd", lr_sgd)] #PR Score at recall of 50 is:0.7972350230414746
    #models = [('xgb', xgb), ('et', et), ('sgd', svm_sgd)] #PR Score at recall of 50 is: 0.8309317963496637
    #models = [('xgb', xgb), ('et', et), ('sgd', svm_sgd), ("knn", knn)] #PR Score at recall of 50 is:0.8084112149532711
    #models = [('xgb', xgb), ('et', et), ('sgd', svm_sgd), ("mlp", mlp)] #PR Score at recall of 50 is:0.8114446529080676
    #models = [('xgb', xgb), ('et', et), ('sgd', svm_sgd), ('mlp', mlp)] #0.8024118738404453
    #models = [('xgb', xgb), ('sgd', svm_sgd), ('mlp', mlp)] #0.8024118738404453

    list_pr = []
    #vc = VotingClassifier(estimators=models, voting='soft')

    for interation in range(0, 10):
        Classifier = DecisionTree(data, kfold, xgb, log=True)
        list_pr.append(Classifier.p_at_r_50)

    print(list_pr)
    np.savetxt("Presision_at_50_recall_interative_test.csv", list_pr, delimiter=",")


    #Classifier1 = DecisionTree(data, kfold, XGBClassifier(use_label_encoder=False))
    #Classifier = DecisionTree(data, kfold, BaggingClassifier(n_estimators=100, random_state=0))
    #Classifier = DecisionTree(data, kfold, GradientBoostingClassifier())
    #Classifier = DecisionTree(data, kfold, tree.DecisionTreeClassifier())
    #Classifier = DecisionTree(data, kfold, model)

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