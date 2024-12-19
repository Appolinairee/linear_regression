import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class DataPreprocessor:
    def __init__(self, log_transform=True):
        self.scaler = StandardScaler()

    def impute_missing_values(self, data, columns_median, columns_mean):
        
        
        return data

    def apply_log_transformation(self, data):
        for col in data.columns:
            if np.abs(data[col].skew()) > 0.3:
                data[col] = np.log1p(data[col])
        
        return data

    def scale_data(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled

    def preprocess(self, data, X_train, X_test, columns_median, columns_mean):
        data = self.impute_missing_values(data, columns_median, columns_mean)
        data = self.apply_log_transformation(data)
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        return data, X_train_scaled, X_test_scaled