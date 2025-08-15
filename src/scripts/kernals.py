""" """

import numpy as np
from sklearn.base import BaseEstimator


# Dynamic time warping stuff
def d_DTW(x, x2, dist):
    t1, t2 = len(x), len(x2)

    # Edge Cases

    if len(x) == 0 and len(x2) == 0:
        return 0.0
    elif len(x) == 0 or len(x2) == 0:
        return np.infty

    dp = np.empty((t1 + 1, t2 + 1))
    dp[0, 0] = 0

    for i in range(1, t1 + 1):
        dp[i, 0] = np.infty

    for j in range(1, t2 + 1):
        dp[0, j] = np.infty

    # Standard Procedure

    for i in range(1, t1 + 1):
        for j in range(1, t2 + 1):
            d = dist(x[i - 1], x2[j - 1])
            gamma = d + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
            dp[i, j] = gamma

    return dp[t1, t2]


def d1(x, x2):
    if x != x2:
        return 1
    else:
        return 0


def d2(x, x2):
    return np.array(x) - np.array(x2) ** 2


def d3(x, x2):
    return abs(x - x2)


def map_func(x):
    t_something = len(x)
    delta = np.zeros(t_something)
    for i in range(1, t_something):
        if i == 0:
            continue
        if i == t_something - 1:
            delta[i] = x[i] - x[i - 1]
        else:
            delta[i] = 0.5 * (x[i] - x[i - 1] + ((x[i + 1] - x[i - 1]) / 2))

    return delta


def d4(x, x2):
    del_x = map_func(x)
    del_x2 = map_func(x2)
    dists = []
    for i in range(len(x)):
        dists[i] = d2(x[i], x2[i])
    return dists


def d_euclidean(x, x2):
    return np.linalg.norm(np.array(x) - np.array(x2))


k1_hyp, k2_hyp, k3_hyp = [
    lambda lmbd: (lambda x, x2: np.exp(-lmbd * d_DTW(x, x2, d))) for d in [d1, d2, d3]
]

k_rbf = lambda gamma: (lambda x, x2: np.exp(-gamma * (d_euclidean(x, x2) ** 2)))

k_poly = lambda p: (lambda x, x2: np.dot(x.T, x2) ** p)


def build_dtw_gram_matrix(xs, x2s, k):
    """
    xs: collection of sequences (vectors of possibly varying length)
    x2s: the same, needed for prediction
    k: a kernel function that maps two sequences of possibly different length to a real
    The function returns the Gram matrix with respect to k of the data xs.
    """
    t1, t2 = len(xs), len(x2s)
    K = np.empty((t1, t2))

    for i in range(t1):
        for j in range(i, t2):
            K[i, j] = k(xs[i], x2s[j])
            if i < t2 and j < t1:
                K[j, i] = K[i, j]

    return K


def L2_reg(w, lbda):
    return 0.5 * lbda * (np.dot(w.T, w)), lbda * w


def hinge_loss(h, y):
    n = len(h)
    l = np.maximum(0, np.ones(n) - y * h)
    g = -y * (h > 0)
    return l, g


def learn_reg_kernel_ERM(
    X,
    y,
    lbda,
    k,
    loss=hinge_loss,
    reg=L2_reg,
    max_iter=200,
    tol=0.001,
    eta=1.0,
    verbose=False,
):
    """Kernel Linear Regression (default: kernelized L_2 SVM)
    X -- data, each row = instance
    y -- vector of labels, n_rows(X) == y.shape[0]
    lbda -- regularization coefficient lambda
    k -- the kernel function
    loss -- loss function, returns vector of losses (for each instance) AND the gradient
    reg -- regularization function, returns reg-loss and gradient
    max_iter -- max. number of iterations of gradient descent
    tol -- stop if norm(gradient) < tol
    eta -- learning rate
    """

    g_old = None

    K = build_dtw_gram_matrix(
        X, X, k
    )  # MODIFY; fill in; hint: use gram matrix defined above
    w = np.random.randn(
        K.shape[1]
    )  # modify; hint: The matrix K should be used and w has as many entries as training examples.

    for t in range(max_iter):
        h = np.dot(K, w)  # MODIFY; hint: see slide 20,21, and 35 (primal vs. dual view)
        l, lg = loss(h, y)

        if verbose:
            print("training loss: " + str(np.mean(l)))

        r, rg = reg(w, lbda)
        g = lg + rg

        if g_old is not None:
            eta = 0.9**t

        w = w - eta * g
        if np.linalg.norm(eta * g) < tol:
            break
        g_old = g

    return w, K


def predict(alpha, X, X_train, k):
    K = build_dtw_gram_matrix(X_train, X, k)
    y_pred = np.dot(K, alpha)
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1

    return y_pred


class KernelEstimator(BaseEstimator):

    def __init__(self, k, lbda, max_iter = 20000, eta = 1, tol = 1e-3):
        self.k = k
        self.lbda = lbda
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol

    def fit(self, X, y):
        self._X_train = X
        self._alpha, _ = learn_reg_kernel_ERM(
            X, y, lbda=self.lbda, k=self.k, max_iter=self.max_iter, eta=self.eta, tol=self.tol
        )
        return self

    def predict(self, X):
        return predict(self._alpha, self._X_train, X, self.k)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
