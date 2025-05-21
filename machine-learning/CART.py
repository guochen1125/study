import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer



if __name__=="__main__":
    data = {
        'Age': [25, 30, np.nan, 35, 40, 45, np.nan, 50],
        'Gender': ['Male', 'Female', 'Female', np.nan, 'Male', 'Female', 'Male', np.nan],
        'Income': [50000, 60000, 55000, 65000, np.nan, 70000, 62000, 72000],
        'Education': ['Bachelor', 'Master', 'High School', 'Bachelor', np.nan, 'Master', 'High School', 'Bachelor'],
        'SpendingScore': [60, 70, 65, 80, 75, 85, 78, 90]
    }
    df = pd.DataFrame(data)

    numeric_features=['Age','Income']
    categorical_features = ['Gender', 'Education']