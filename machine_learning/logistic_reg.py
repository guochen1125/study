import numpy as np


def compute_cost(X, y, w, b, lam):
    y_pred = 1 / (1 + np.exp(-(X @ w + b)))
    cost = -1 * np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    if lam != 0:
        cost += lam / 2 / X.shape[0] * np.sum(w**2)
    return cost


def compute_gradient(y_pred, y, X, lam, w):

    g_w = np.mean((y_pred - y) * X, 0).reshape(-1, 1)
    g_b = np.mean(y_pred - y)
    if lam != 0:
        g_w += lam / X.shape[1] * w
    return g_w, g_b


def gradient_descent(X_train, y_train, w, b, num_iters, alpha):
    cost_history = []
    for i in range(num_iters):
        y_pred = 1 / (1 + np.exp(-(X_train @ w + b)))
        cost_history.append(compute_cost(y_pred, y_train))
        g_w, g_b = compute_gradient(y_pred, y_train, X_train,w)
        w -= alpha * g_w
        b -= alpha * g_b
    return w, b, y_pred


if __name__ == "__main__":

    np.random.seed(1)
    X_tmp = np.random.rand(5, 6)
    y_tmp = np.array([0, 1, 0, 1, 0])
    w_tmp = (
        np.random.rand(6).reshape(
            -1,
        )
        - 0.5
    )
    b_tmp = 0.5
    lambda_tmp = 0.7
    cost_tmp = compute_cost(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

    print("Regularized cost:", cost_tmp)
    exit()

    X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

    n, m = X_train.shape

    w_init = np.array([0.0, 0.0]).reshape(-1, 1)
    b_init = 0.0

    w, b, y_pred = gradient_descent(
        X_train, y_train, w_init, b_init, num_iters=10000, alpha=0.1
    )

    print(w, b)
    print(y_pred)
