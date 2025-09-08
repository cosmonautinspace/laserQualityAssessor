"""
Script for Kernel methods mixed-in with Sklearn's SVC
Provides classes for DTW, RBF and Polynomial Kernals

NOTE: The SVC class by default supports Linear, Polynomial and RBF Kernels,
But the choice was made to rewrite everything,
as it's easier to explain when you can see the \"low level\" implementation while presenting
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from dtaidistance import dtw


class dtw_SVC(BaseEstimator, ClassifierMixin):
    """
    Class for building a DTW SVM

    :param C: (float) Coefficient of regularization
    :param gamma: (float) Smoothing coefficient for DTW
    :param fast: (bool) Compute dtw distance in C
    :param inner_dist: (string) Pointwise distance used for DTW
    """

    def __init__(self, C=1.0, gamma=0.01, fast=True, inner_dist="euclidean"):
        self.C = C
        self.gamma = gamma
        self.svc_ = None
        self.X_train_ = None
        self.dist_mat_ = None
        self.fast = fast
        self.inner_dist = inner_dist

    def _build_gram_matrix(self, X, Y=None):
        if Y is None:
            Y = X

        t1, t2 = len(X), len(Y)
        D = np.zeros((t1, t2), dtype=np.float64)

        if self.fast:
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

    def decision_function(self, X):
        K_predict = self._build_gram_matrix(X, self.X_train_)
        return self.svc_.decision_function(K_predict)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class rbf_SVC(BaseEstimator, ClassifierMixin):
    """
    Class for building a RBF SVM

    :param C: (float) Coefficient of regularization
    :param gamma: (float) Smoothing coefficient for RBF
    """

    def __init__(self, C=1.0, gamma=0.01):
        self.C = C
        self.gamma = gamma
        self.svc_ = None
        self.X_train_ = None
        self.dist_mat_ = None

    def _dist_euclidean(self, X, Y):
        return np.linalg.norm(np.array(X) - np.array(Y))

    def _build_gram_matrix(self, X, Y=None):
        if Y is None:
            Y = X

        t1, t2 = len(X), len(Y)
        D = np.zeros((t1, t2), dtype=np.float64)

        for i in range(t1):
            for j in range(t2):
                D[i, j] = (self._dist_euclidean(X[i], Y[j])) ** 2

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

    def decision_function(self, X):
        K_predict = self._build_gram_matrix(X, self.X_train_)
        return self.svc_.decision_function(K_predict)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class poly_SVC(BaseEstimator, ClassifierMixin):
    """
    Class for building a Polynomial SVM

    :param C: (float) Coefficient of regularization
    :param degree: (int) degree of polynomial
    """

    def __init__(self, C=1.0, degree=2):
        self.C = C
        self.degree = degree
        self.svc_ = None
        self.X_train_ = None
        self.dist_mat_ = None

    def _build_gram_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        """
        t1, t2 = len(X), len(Y)
        D = np.zeros((t1,t2), dtype=np.float64)

        for i in range(t1):
            for j in range(t2):
                D[i,j] = np.dot(X[i], Y[j]) + 1
           
        K = D ** self.degree
        return K
        """
        return (np.dot(X, Y.T) + 1) ** self.degree

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

    def decision_function(self, X):
        K_predict = self._build_gram_matrix(X, self.X_train_)
        return self.svc_.decision_function(K_predict)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
