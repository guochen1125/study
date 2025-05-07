from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


def compute_y_pred(X, w, b):
    return X.dot(w) + b


if __name__ == "__main__":
    datasets = fetch_california_housing()
    X, y = datasets.data, datasets.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(X_train.shape)

    w_init = np.ones(X_train.shape[1]).reshape(-1, 1)
    b_init = 1
    y_pred = compute_y_pred(X_train, w_init, b_init)
    print(y_pred.shape)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    print(score)
