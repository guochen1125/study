import numpy as np
from sklearn.linear_model import LogisticRegression


X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(y_pred, model.score(X, y))
