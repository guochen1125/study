import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.python.keras.activations import sigmoid

# ----------------- linear ---------------------
# X_train = np.array([[1.0], [2.0]], dtype=np.float32)
# Y_train = np.array([[300.0], [500.0]], dtype=np.float32)

# linear_layer = Dense(units=1, activation="linear")
# # linear_layer(X_train)  # 自动构建（或者用下面的方式）
# linear_layer.build((None, 1))  # 指明输入 shape 是 [batch_size, 1]

# w = np.array([[200.0]], dtype=np.float32)
# b = np.array([100.0], dtype=np.float32)
# linear_layer.set_weights([w, b])

# pred_tf = linear_layer(X_train)
# print(pred_tf)

# ----------------- logistic ----------------------
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

log_model=Sequential([Dense(1, input_dim=1,  activation = 'sigmoid', name='Layer1')])
logistic_layer_1=log_model.get_layer('Layer1')

w = np.array([[2]])
b = np.array([-4.5])
logistic_layer_1.set_weights([w,b])
log_pred=log_model(X_train[0].reshape(1,1))
print(log_pred)