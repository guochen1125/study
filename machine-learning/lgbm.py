from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from lightgbm import LGBMClassifier

X, y = make_classification(n_samples=500, n_features=15, random_state=666)
model=LGBMClassifier()