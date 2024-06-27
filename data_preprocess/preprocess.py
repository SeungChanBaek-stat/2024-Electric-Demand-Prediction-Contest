import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys, os
sys.path.append(os.pardir)
from data_preprocess.functions import seasonal_transform, k_fold_mean_target_encoding, k_fold_cv_alpha
from data_preprocess.functions import clean_data, expanding_mean_target_encoding
import seaborn as sns


# CSV 파일 경로
data_dir = "c:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\data"
train_path = data_dir + "\\electric_train.csv"
test_path = data_dir + "\\electric_test.csv"

# CSV 파일을 데이터프레임으로 읽기
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
print(train_data.shape)



# # # 기존 regular 변환
# train_data = train_data[train_data['electric_train.nph_rn_60m'] >= 0]
# train_data['electric_train.nph_rn_60m'] = np.log1p(train_data['electric_train.nph_rn_60m'])
# train_data = train_data[train_data['electric_train.nph_ws_10m'] >= 0]
# train_data['electric_train.nph_ws_10m'] = np.log1p(train_data['electric_train.nph_ws_10m'])
# train_data['electric_train.tm'] = pd.to_datetime(train_data['electric_train.tm'])



# # train_data에서 'electric_train.num' == 17677 인 갯수 확인
# print(train_data[train_data['electric_train.num'] == 17677].shape)

# # train_data에서 'electric_train.num' == 17677 인 행 삭제
# train_data = train_data[train_data['electric_train.num'] != 17677]
# print(train_data[train_data['electric_train.num'] == 17677].shape)


# # test_data에서 'NUM' == 17677 열이 존재하는지 확인
# print(test_data[test_data['NUM'] == 17677].shape)








mean = np.mean(train_data['electric_train.elec'])
std_dev = np.std(train_data['electric_train.elec'])

outlier_condition = (train_data['electric_train.elec'] > mean + 5 * std_dev) | (train_data['electric_train.elec'] < mean - 5 * std_dev)

outliers = train_data[outlier_condition]

train_data = train_data[~outlier_condition]
non_outliers = train_data[~outlier_condition]   

print("Number of outliers : ", outliers.shape[0])
print("Number of non-outliers : ", train_data.shape[0])

max_non_outliers = train_data['electric_train.elec'].max()
min_non_outliers = train_data['electric_train.elec'].min()

print("Max non-outliers : ", max_non_outliers)
print("Min non-outliers : ", min_non_outliers)

max_outliers = outliers['electric_train.elec'].max()
min_outliers = outliers['electric_train.elec'].min()

print("Max non-outliers : ", max_outliers)
print("Min non-outliers : ", min_outliers)



print(train_data.shape)






'nph_ta', 'nph_hm', 'nph_rn_60m', 'nph_ws_10m'

# 이상치 제거
train_data = clean_data(train_data, 6)
print(f'train_data shape: {train_data.shape}')


# test data 변수명 소문자로 바꾸기
test_data.columns = test_data.columns.str.lower()
# test data 변수명앞에 electric_test. 붙이기
test_data.columns = 'electric_test.' + test_data.columns


# 필요없는 변수 삭제
train_data.drop(['electric_train.n', 
               'electric_train.sum_load', 
               'electric_train.n_mean_load', 
               'electric_train.sum_qctr',
               'electric_train.stn',
               'electric_train.hh24'], axis=1, inplace=True)
test_data.drop(['electric_test.hh24',
                'electric_test.stn'], axis=1, inplace=True)




# 훈련 및 테스트 데이터 변수명 통일
train_data.columns = train_data.columns.str.replace('electric_train.', 'electric.')
test_data.columns = test_data.columns.str.replace('electric_test.', 'electric.')

# train_data에서 'electric.tm' 변수를 연도, 월, 일, 시간 변수로 분리
train_data['electric.tm'] = pd.to_datetime(train_data['electric.tm'])
train_data['year'] = train_data['electric.tm'].dt.year
train_data['month'] = train_data['electric.tm'].dt.month
train_data['day_of_year'] = train_data['electric.tm'].dt.dayofyear
train_data['hour'] = train_data['electric.tm'].dt.hour
train_data.drop(['electric.tm'], axis=1, inplace=True)
print(len(train_data.columns))
print(train_data.columns)

# test_data에서 'electric.tm' 변수를 연도, 월, 일, 시간 변수로 분리
test_data['electric.tm'] = pd.to_datetime(test_data['electric.tm'])
test_data['year'] = test_data['electric.tm'].dt.year
test_data['month'] = test_data['electric.tm'].dt.month
test_data['day_of_year'] = test_data['electric.tm'].dt.dayofyear
test_data['hour'] = test_data['electric.tm'].dt.hour
test_data.drop(['electric.tm'], axis=1, inplace=True)
print(len(test_data.columns))
print(test_data.columns)








# 데이터 전처리
continuous_features = ['electric.nph_ta', 
                       'electric.nph_hm', 'electric.nph_ws_10m', 
                       'electric.nph_rn_60m', 'electric.nph_ta_chi']
categorical_features = ['electric.num', 'electric.week_name']
seasonal_features = ['month', 'day_of_year', 'hour', 'electric.weekday']
total_features = continuous_features + categorical_features + seasonal_features
print(f'total_features: {len(total_features)}')
print(f'train_data.columns: {len(train_data.columns)}') # elec과 year변수가 추가되어있음

scaler = MinMaxScaler()
train_data[continuous_features] = scaler.fit_transform(train_data[continuous_features].astype(np.float32))
train_data = seasonal_transform(train_data, seasonal_features)

test_data[continuous_features] = scaler.transform(test_data[continuous_features].astype(np.float32))
test_data = seasonal_transform(test_data, seasonal_features)


# Example usage
target_col = 'electric.elec'
column_to_encode = 'electric.num'

# Apply K-Fold Mean Target Encoding
train_data['electric.num_encoded'], test_data['electric.num_encoded'] = k_fold_mean_target_encoding(train_data, test_data, column_to_encode, target_col, alpha = 0.01)


# Apply Expanding Mean Target Encoding
# train_data['electric.num_encoded'], test_data['electric.num_encoded'] = expanding_mean_target_encoding(train_data, test_data, column_to_encode, target_col, alpha=0.2)

print("train_encoded")
print(train_data.shape)
print(train_data.sample(10))
print(train_data.columns)

print("test_encoded")
print(test_data.shape)
print(test_data.sample(10))
print(test_data.columns)




# daily_elec_mean_after = train_data.groupby('electric.tm')['electric.elec'].mean()

# # 일일 전력 사용량의 평균값을 순서대로 시각화
# plt.figure(figsize=(20, 5))
# plt.plot(daily_elec_mean_after)
# plt.title('Daily Mean after outlier deletion - Electricity Usage')
# plt.show()





# 전처리가 끝났으므로 필요없는 변수 삭제
train_data.drop(['electric.weekday', 'electric.week_name',
                 'month', 'day_of_year', 'hour',], axis=1, inplace=True)
test_data.drop(['electric.weekday', 'electric.week_name',
                'month', 'day_of_year', 'hour',], axis=1, inplace=True)

print(train_data.shape)
print(train_data.columns)
print(test_data.shape)
print(test_data.columns)




# 이상치가 제거및 전처리가 완료된 데이터셋 저장
train_data.to_csv(os.path.join(data_dir, 'electric_train_preprocessed.csv'), index=False)
test_data.to_csv(os.path.join(data_dir, 'electric_test_preprocessed.csv'), index=False)
print("전처리된 데이터셋 저장 완료")










