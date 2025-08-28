import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from scripts.kernals import d_DTW, d2
from dtaidistance import dtw

class dtw_SVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, gamma=0.01, estimate=False):
        self.C = C
        self.gamma = gamma
        self.svc_ = None
        self.X_train_ = None
        self.dist_mat_ = None
        self.estimate = estimate

    def fit(self, X, y):
        self.X_train_ = X
        if self.estimate:
            self.dist_mat_ = dtw.distance_matrix_fast(X)
        else:
            self.dist_mat_ = dtw.distance_matrix(X)
        self.dist_mat_ = np.array(self.dist_mat_, dtype=np.float64)
        K_train = np.exp(-self.gamma * self.dist_mat_)
        
        self.svc_ = SVC(C=self.C, kernel="precomputed") 
        self.svc_.fit(K_train, y)
        return self
    
    def predict(self, X):
        if self.estimate:
            D_predict = dtw.distance_matrix_fast(X, self.X_train_)
        else:
            D_predict = dtw.distance_matrix(X, self.X_train_)
        D_predict = np.array(D_predict, dtype=np.float64)
        K_predict = np.exp(-self.gamma * D_predict)
        return self.svc_.predict(K_predict)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)