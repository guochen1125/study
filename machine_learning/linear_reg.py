from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import zscore


# def zscore(X):
#     return (X - np.mean(X, 0)) / np.std(X, 0)


def compute_y_pred(X, w, b):
    return X @ w + b


def compute_cost(y_pred, y):
    return np.mean((y_pred - y) ** 2) / 2


def compute_gradient(y_pred, y, X):
    g_w = np.mean((y_pred - y) * X, 0).reshape(-1, 1)
    g_b = np.mean(y_pred - y)
    return g_w, g_b


def draw_cost_iter(cost_history, save_pth):
    # 绘制 cost 关于迭代次数 i 的散点图
    plt.scatter(range(len(cost_history)), cost_history, color="blue", marker="o")
    plt.title("Cost vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig(save_pth)


def gradient_descent(X_train, y_train, w, b, num_iters, alpha):
    cost_history = []
    for i in range(num_iters):
        y_pred = compute_y_pred(X_train, w, b)
        cost_history.append(compute_cost(y_pred, y_train))
        g_w, g_b = compute_gradient(y_pred, y_train, X_train)
        w -= alpha * g_w
        b -= alpha * g_b
    return y_pred, cost_history


if __name__ == "__main__":
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    # X_train = zscore(X_train, 0)

    y_train = np.array([460, 232, 178]).reshape(-1, 1)
    w_init = np.zeros(4).reshape(-1, 1)
    b_init = 0

    n, m = X_train.shape  # n：样本个数 m：特征个数

    y_pred, cost_history = gradient_descent(
        X_train, y_train, w_init, b_init, num_iters=1000, alpha=5.0e-7
    )
    draw_cost_iter(cost_history[100:], "cost_iter.png")

    print(y_pred)
