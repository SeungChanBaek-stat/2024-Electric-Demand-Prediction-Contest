import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import os
import time
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from numpy.random import normal
from sklearn.model_selection import StratifiedKFold, KFold



# 범주형 변수 원핫 인코딩 및 계절적 변수 변환
def seasonal_transform(data, features):
    sin_cos_features = []
    for feature in features:
        sin_cos_features.append(pd.DataFrame({
            feature + '_sin': np.sin(2 * np.pi * data[feature] / data[feature].max()).astype(np.float32),
            feature + '_cos': np.cos(2 * np.pi * data[feature] / data[feature].max()).astype(np.float32)
        }))
    result = pd.concat([data] + sin_cos_features, axis=1)
    return result


def k_fold_mean_target_encoding(train_data, test_data, column, target_col, n_splits=5, alpha=0.2):
    """
    K-Fold Mean Target Encoding
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    train_encoded = pd.Series(np.zeros(train_data.shape[0], dtype=np.float32), index=train_data.index)

    for train_index, val_index in kf.split(train_data):
        X_train, X_val = train_data.iloc[train_index], train_data.iloc[val_index]
        means = X_train.groupby(column)[target_col].mean()
        counts = X_train.groupby(column)[target_col].count()
        smooth = (means * counts + train_data[target_col].mean() * alpha) / (counts + alpha)
        train_encoded.iloc[val_index] = train_data[column].iloc[val_index].map(smooth).astype(np.float32)
    
    overall_mean = train_data[target_col].mean()
    train_encoded.fillna(overall_mean, inplace=True)

    # Encoding the test data using the whole train data
    means = train_data.groupby(column)[target_col].mean()
    counts = train_data.groupby(column)[target_col].count()
    smooth = (means * counts + overall_mean * alpha) / (counts + alpha)
    test_encoded = test_data[column].map(smooth).astype(np.float32)
    test_encoded.fillna(overall_mean, inplace=True)

    return train_encoded, test_encoded




def expanding_mean_target_encoding(train_data, test_data, column, target_col, alpha=0.2):
    """
    Expanding Mean Target Encoding
    """
    train_encoded = pd.Series(np.zeros(train_data.shape[0], dtype=np.float32), index=train_data.index)
    overall_mean = train_data[target_col].mean()

    # Initialize the cumulative sum and count
    cumsum = train_data.groupby(column)[target_col].cumsum() - train_data[target_col]
    cumcount = train_data.groupby(column).cumcount()

    # Calculate the expanding mean with smoothing
    train_encoded = (cumsum + overall_mean * alpha) / (cumcount + alpha)
    
    # Encoding the test data using the whole train data
    means = train_data.groupby(column)[target_col].mean()
    counts = train_data.groupby(column)[target_col].count()
    smooth = (means * counts + overall_mean * alpha) / (counts + alpha)
    test_encoded = test_data[column].map(smooth).astype(np.float32)
    test_encoded.fillna(overall_mean, inplace=True)

    return train_encoded, test_encoded







def k_fold_cv_alpha(train_data, column, target_col, alphas, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    best_alpha = None
    best_score = float('inf')
    results = []

    for alpha in alphas:
        fold_scores = []
        for train_index, val_index in kf.split(train_data):
            train_fold = train_data.iloc[train_index]
            val_fold = train_data.iloc[val_index]
            
            _, encoded = k_fold_mean_target_encoding(train_fold, val_fold, column, target_col, n_splits, alpha)
            # _, encoded = expanding_mean_target_encoding(train_fold, val_fold, column, target_col, alpha)
            val_fold.loc[:, f'{column}_encoded'] = encoded
            
            mse = mean_squared_error(val_fold[target_col], val_fold[f'{column}_encoded'])
            fold_scores.append(mse)
        
        avg_score = np.mean(fold_scores)
        results.append((alpha, avg_score))
        
        if avg_score < best_score:
            best_score = avg_score
            best_alpha = alpha

    return best_alpha, best_score, results






# 이상치 제거 함수
def remove_extreme_outliers(df, col, sigma):
    mean = np.mean(df[col])
    std_dev = np.std(df[col])
    return df[(df[col] > mean - sigma * std_dev) & (df[col] < mean + sigma * std_dev)]

# 이상치 제거 및 음수 값 제거
def clean_data(data, sigma):
    cleaned_data = data.copy()
    for feature in ['nph_ta', 'nph_hm', 'nph_rn_60m', 'nph_ws_10m']:
        feature_col = f'electric_train.{feature}'
        cleaned_data = remove_extreme_outliers(cleaned_data, feature_col, sigma)
        # if feature == 'nph_rn_60m':
        #     cleaned_data = remove_extreme_outliers(cleaned_data, feature_col, sigma)
        # elif feature == 'nph_ws_10m':
        #     cleaned_data = remove_extreme_outliers(cleaned_data, feature_col, sigma)
    return cleaned_data



