from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class Model(ABC):

    @abstractmethod
    def _search_best_params(self,X,y,cv, score_metric):
        """Search best hyperparams"""
        pass


    @abstractmethod
    def fit(self, X, y, feature_list = [], cv=5, score_metric='f1'):
        """Training algorithm for the machine learning model"""
        pass


    @abstractmethod
    def transform(self, x):
        """Prediction of new data"""
        pass


class TreeModel(Model):
    def __init__(self, grid_params = None, scale_input = False):
        if grid_params is None:
            self.grid_params = {'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4]}
        else:
            self.grid_params = grid_params
        
        self.scale_input = scale_input
        return
    

    def _search_best_params(self,X,y,cv, score_metric):
        mdl = GridSearchCV(DecisionTreeClassifier(), self.grid_params, cv=cv,scoring=score_metric) 
        mdl.fit(X, y)
        return mdl.best_params_
        

    def fit(self, X, y, feature_list = [], cv=5, score_metric='f1'):
        if len(feature_list) == 0:
            feature_list = X.columns
        self.feature_list = feature_list

        if self.scale_input:
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X[feature_list])
        else:
            X_train = X[feature_list]
        
        
        params = self._search_best_params(X_train,y,cv, score_metric)
        mdl = DecisionTreeClassifier(**params)
        mdl.fit(X_train, y)
        self.mdl = mdl
        return mdl
    

    def transform(self, X):
        if self.scale_input:
            X_pred = self.scaler.transform(X[self.feature_list])
        else:
            X_pred = X[self.feature_list]
        y_pred = self.mdl.predict_proba(X_pred)
        return y_pred


class SVMModel(Model):
    def __init__(self, grid_params = None, scale_input = False):
        if grid_params is None:
            self.grid_params = {'C': [0.1, 1, 10],
                                'gamma': [1, 0.1, 0.01],
                                'kernel': ['rbf', 'linear']
                                }
        else:
            self.grid_params = grid_params
        
        self.scale_input = scale_input
        return
    

    def _search_best_params(self,X,y,cv, score_metric):
        mdl = GridSearchCV(SVC(), self.grid_params, cv=cv,scoring=score_metric) 
        mdl.fit(X, y)
        return mdl.best_params_
        

    def fit(self, X, y, feature_list = [], cv=5, score_metric='f1'):
        if len(feature_list) == 0:
            feature_list = X.columns
        self.feature_list = feature_list

        if self.scale_input:
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X[feature_list])
        else:
            X_train = X[feature_list]
        
        
        params = self._search_best_params(X_train,y,cv, score_metric)
        mdl = SVC(**params)
        mdl.fit(X_train, y)
        self.mdl = mdl
        return mdl
    

    def transform(self, X):
        if self.scale_input:
            X_pred = self.scaler.transform(X[self.feature_list])
        else:
            X_pred = X[self.feature_list]
        y_pred = self.mdl.predict_proba(X_pred)
        return y_pred


class KNNModel(Model):
    def __init__(self, grid_params = None, scale_input = False):
        if grid_params is None:
            self.grid_params = {"n_neighbors":list(range(1,31)), "weights":["uniform", "distance"]}
        else:
            self.grid_params = grid_params
        
        self.scale_input = scale_input
        return
    

    def _search_best_params(self,X,y,cv, score_metric):
        mdl = GridSearchCV(KNeighborsClassifier(), self.grid_params, cv=cv,scoring=score_metric) 
        mdl.fit(X, y)
        return mdl.best_params_
        

    def fit(self, X, y, feature_list = [], cv=5, score_metric='f1'):
        if len(feature_list) == 0:
            feature_list = X.columns
        self.feature_list = feature_list

        if self.scale_input:
            self.scaler = MinMaxScaler()
            X_train = self.scaler.fit_transform(X[feature_list])
        else:
            X_train = X[feature_list]
        
        
        params = self._search_best_params(X_train,y,cv, score_metric)
        mdl = KNeighborsClassifier(**params)
        mdl.fit(X_train, y)
        self.mdl = mdl
        return mdl
    

    def transform(self, X):
        if self.scale_input:
            X_pred = self.scaler.transform(X[self.feature_list])
        else:
            X_pred = X[self.feature_list]
        y_pred = self.mdl.predict_proba(X_pred)
        return y_pred
        





