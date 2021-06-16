import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score


class model_finder:
    """

    """
    def __init__(self):
        self.gnb = GaussianNB()
        self.rf = RandomForestClassifier(n_jobs=-1)
        self.svm = SVC(gamma='auto')

    def get_best_param_for_naive_bays(self,X_train,X_test,y_train,y_test):
        """

        :param X_train:
        :param y_train:
        :return:
        """
        #self.y_train = y_train.values.ravel()
        #self.y_test = y_test.values.ravel()
        #self.y_train = np.array(self.train_y).astype(int)
        #self.y_test = np.array(self.test_y).astype(int)
        try:
            self.param_grid = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}
            self.grid = RandomizedSearchCV(estimator=self.gnb, param_distributions=self.param_grid, cv=5, verbose=2)
            self.grid.fit(X_train,y_train)

            # Extract best parameters
            self.best_param =self.grid.best_params_['var_smoothing']

            # new model with best parameters
            self.gnb =GaussianNB(var_smoothing=self.best_param)
            self.gnb.fit(X_train,y_train)
            self.prediction = self.gnb.predict(X_test)
            self.accuracy = accuracy_score(y_test.values,self.prediction)
            self.auc_score = roc_auc_score(y_test.values,self.prediction)
            return self.accuracy,self.auc_score
            #print("Naive bays gives accuracy {} and AUC score {} ".format(np.round(self.accuracy,4),np.round(self.auc_score,4)))
        except:
            print("Error Occured in Naive Bays model tuning")


    def get_best_params_for_random_forest(self,X_train,X_test,y_train,y_test):
        """

        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        try:
            self.param_grid_rf = {'max_depth': [5,10,11,12,14,15, None],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [10,100, 180, 230]}
            self.grid =RandomizedSearchCV(estimator=self.rf,param_distributions=self.param_grid_rf,cv=3,n_jobs=-1)
            self.grid.fit(X_train, y_train)

            # Extract best parameters
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.min_samples_leaf = self.grid.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # new model with best parameters
            self.rf = RandomForestClassifier(n_estimators=self.n_estimators,max_features=self.max_features,min_samples_leaf=self.min_samples_leaf,
                                  min_samples_split=self.min_samples_split,max_depth=self.max_depth)
            self.rf.fit(X_train, y_train)
            self.prediction = self.rf.predict(X_test)
            self.accuracy = accuracy_score(y_test.values, self.prediction)
            self.auc_score = roc_auc_score(y_test.values, self.prediction)
            return self.accuracy, self.auc_score
            #print("RandomForest gives accuracy {} and AUC score {} ".format(np.round(self.accuracy, 4),
                      #                                                    np.round(self.auc_score, 4)))

        except:
            print("Error Occured in RandomForest model tuning")


    def get_best_params_of_SVM(self,X_train,X_test,y_train,y_test):
        """

        :param X_train:
        :param X_test:
        :param y_train:
        :param y_test:
        :return:
        """
        try:
            self.param_grid_svm = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                                  "kernel": ['rbf', 'poly', 'sigmoid'],
                                  "shrinking": [True, False]}
            self.grid =RandomizedSearchCV(estimator=self.svm,param_distributions=self.param_grid_svm,cv=3,n_jobs=-1)
            self.grid.fit(X_train, y_train)

            # Extract best parameters
            self.C = self.grid.best_params_['C']
            self.kernel = self.grid.best_params_['kernel']
            self.shrinking = self.grid.best_params_['shrinking']

            # new model with best parameters
            self.svm = SVC(C=self.C,kernel=self.kernel,shrinking=self.shrinking,gamma='auto')
            self.svm.fit(X_train, y_train)
            self.prediction = self.svm.predict(X_test)
            self.accuracy = accuracy_score(y_test.values, self.prediction)
            self.auc_score = roc_auc_score(y_test.values, self.prediction)
            return self.accuracy, self.auc_score
            #print("SVM gives accuracy {} and AUC score {} ".format(np.round(self.accuracy, 4),
                                                                         # np.round(self.auc_score, 4)))

        except:
            print("Error Occured in SVM model tuning")

    def get_best_Model(self,train_x,train_y,test_x,test_y):
        """

        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :return:
        """
        try:
            self.nb_accuracy,self.nb_auc =self.get_best_param_for_naive_bays(train_x,train_y,test_x,test_y)
            self.rf_accuracy,self.rf_auc = self.get_best_params_for_random_forest(train_x,train_y,test_x,test_y)
            self.svm_accuracy,self.svm_auc = self.get_best_params_of_SVM(train_x,train_y,test_x,test_y)

            if (self.nb_auc >= self.rf_auc) and (self.nb_auc >= self.svm_auc):
                print("Best model is Naive bays with AUC score = {}".format(self.nb_auc))

            elif (self.rf_auc >= self.nb_auc) and (self.rf_auc >= self.svm_auc):
                print("Best model is RandomForest with AUC score = {}".format(self.rf_auc))

            else:
                print("Best model is SVM with AUC score = {}".format(self.svm_auc))

        except:
            print("Error occured in get best model method")