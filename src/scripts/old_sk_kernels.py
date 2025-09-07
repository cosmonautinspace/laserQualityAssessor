import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from scripts.kernals import d_DTW, d2
from dtaidistance import dtw


class dtw_SVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, gamma=0.01, estimate=False, inner_dist="euclidean"):
        self.C = C
        self.gamma = gamma
        self.svc_ = None
        self.X_train_ = None
        self.dist_mat_ = None
        self.estimate = estimate
        self.inner_dist = inner_dist

    def _build_gram_matrix(self, X, Y=None):
        if Y is None:
            Y = X

        t1, t2 = len(X), len(Y)
        D = np.zeros((t1, t2), dtype=np.float64)

        if self.estimate:
            for i in range(t1):
                for j in range(t2):
                    D[i, j] = dtw.distance_fast(X[i], Y[j], inner_dist=self.inner_dist)
        else:
            for i in range(t1):
                for j in range(t2):
                    D[i, j] = dtw.distance(X[i], Y[j], inner_dist=self.inner_dist)

        K = np.exp(-self.gamma * D)
        return K

    def fit(self, X, y):
        self.X_train_ = X
        k_train = self._build_gram_matrix(X)
        self.svc_ = SVC(C=self.C, kernel="precomputed")
        self.svc_.fit(k_train, y)
        self.classes_ = self.svc_.classes_  # forward classes_
        return self

    def predict(self, X):
        k_predict = self._build_gram_matrix(X, self.X_train_)
        return self.svc_.predict(k_predict)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
