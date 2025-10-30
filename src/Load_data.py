# Data loading and preprocessing module

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def Load_data(file_path=None):
    if file_path:
        # Load dataset
        df = pd.read_csv(file_path)

        # Drop ID column (not useful for prediction)
        df = df.drop(columns=['id'], errors='ignore')

        # Convert diagnosis to numeric labels: M = 1, B = 0
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

        # Separate features and target
        X = df.drop(columns=['diagnosis'], errors='ignore')
        y = df['diagnosis']

        # Keep only numeric columns (ignore any string columns if exist)
        X = X.select_dtypes(include=[np.number])

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    else:
        # Load sklearn's built-in dataset if no file provided
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
