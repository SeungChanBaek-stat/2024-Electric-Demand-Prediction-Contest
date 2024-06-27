import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# CSV 파일 경로
data_dir = "c:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\data"
train_path = data_dir + "\\electric_train.csv"
test_path = data_dir + "\\electric_test.csv"

# CSV 파일을 데이터프레임으로 읽기
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# train_df 전체 길이 출력
print(f"Length of train_df: {len(train_df)}")



# electric_train.tm 변수를 시계열로 변환
train_df['electric_train.tm'] = pd.to_datetime(train_df['electric_train.tm'])

# # 데이터프레임 출력
# print(train_df.head())


# 연도 추출
train_df['year'] = train_df['electric_train.tm'].dt.year
# 연도별 데이터 분리
train_2020 = train_df[train_df['year'] == 2020]
train_2021 = train_df[train_df['year'] == 2021]
train_2022 = train_df[train_df['year'] == 2022]





# electric_train.num 열의 유니크한 값 추출
unique_values = train_df['electric_train.num'].unique()

# 유니크한 값의 개수 출력
print("Number of unique values in 'electric_train.num':", len(unique_values))



# 저장 경로 설정
save_dir = "C:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\eda\\uncleaned"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
daily_path_2020 = save_dir + "\\daily\\2020"
if not os.path.exists(daily_path_2020):
    os.makedirs(daily_path_2020)
daily_path_2021 = save_dir + "\\daily\\2021"
if not os.path.exists(daily_path_2021):
    os.makedirs(daily_path_2021)
daily_path_2022 = save_dir + "\\daily\\2022"
if not os.path.exists(daily_path_2022):
    os.makedirs(daily_path_2022)
monthly_path_2020 = save_dir + "\\monthly\\2020"
if not os.path.exists(monthly_path_2020):
    os.makedirs(monthly_path_2020)
monthly_path_2021 = save_dir + "\\monthly\\2021"
if not os.path.exists(monthly_path_2021):
    os.makedirs(monthly_path_2021)
monthly_path_2022 = save_dir + "\\monthly\\2022"
if not os.path.exists(monthly_path_2022):
    os.makedirs(monthly_path_2022)
yearly_path = save_dir + "\\yearly"
if not os.path.exists(yearly_path):
    os.makedirs(yearly_path)
yearly_rm_outlier_path = save_dir + "\\yearly_rm_outlier"
if not os.path.exists(yearly_rm_outlier_path):
    os.makedirs(yearly_rm_outlier_path)



# 일간 저장 경로 설정 2020
nph_ta_daily_save_2020_path = daily_path_2020 + "\\nph_ta"
if not os.path.exists(nph_ta_daily_save_2020_path):
    os.makedirs(nph_ta_daily_save_2020_path)
nph_hm_daily_save_2020_path = daily_path_2020 + "\\nph_hm"
if not os.path.exists(nph_hm_daily_save_2020_path):
    os.makedirs(nph_hm_daily_save_2020_path)
nph_rn_60m_daily_save_2020_path = daily_path_2020 + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_daily_save_2020_path):
    os.makedirs(nph_rn_60m_daily_save_2020_path)
nph_ws_10m_daily_save_2020_path = daily_path_2020 + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_daily_save_2020_path):
    os.makedirs(nph_ws_10m_daily_save_2020_path)

# 일간 저장 경로 설정 2021
nph_ta_daily_save_2021_path = daily_path_2021 + "\\nph_ta"
if not os.path.exists(nph_ta_daily_save_2021_path):
    os.makedirs(nph_ta_daily_save_2021_path)
nph_hm_daily_save_2021_path = daily_path_2021 + "\\nph_hm"
if not os.path.exists(nph_hm_daily_save_2021_path):
    os.makedirs(nph_hm_daily_save_2021_path)
nph_rn_60m_daily_save_2021_path = daily_path_2021 + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_daily_save_2021_path):
    os.makedirs(nph_rn_60m_daily_save_2021_path)
nph_ws_10m_daily_save_2021_path = daily_path_2021 + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_daily_save_2021_path):
    os.makedirs(nph_ws_10m_daily_save_2021_path)

# 일간 저장 경로 설정 2022
nph_ta_daily_save_2022_path = daily_path_2022 + "\\nph_ta"
if not os.path.exists(nph_ta_daily_save_2022_path):
    os.makedirs(nph_ta_daily_save_2022_path)
nph_hm_daily_save_2022_path = daily_path_2022 + "\\nph_hm"
if not os.path.exists(nph_hm_daily_save_2022_path):
    os.makedirs(nph_hm_daily_save_2022_path)
nph_rn_60m_daily_save_2022_path = daily_path_2022 + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_daily_save_2022_path):
    os.makedirs(nph_rn_60m_daily_save_2022_path)
nph_ws_10m_daily_save_2022_path = daily_path_2022 + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_daily_save_2022_path):
    os.makedirs(nph_ws_10m_daily_save_2022_path)


# 월간 저장 경로 설정 2020
nph_ta_monthly_save_2020_path = monthly_path_2020 + "\\nph_ta"
if not os.path.exists(nph_ta_monthly_save_2020_path):
    os.makedirs(nph_ta_monthly_save_2020_path)
nph_hm_monthly_save_2020_path = monthly_path_2020 + "\\nph_hm"
if not os.path.exists(nph_hm_monthly_save_2020_path):
    os.makedirs(nph_hm_monthly_save_2020_path)
nph_rn_60m_monthly_save_2020_path = monthly_path_2020 + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_monthly_save_2020_path):
    os.makedirs(nph_rn_60m_monthly_save_2020_path)
nph_ws_10m_monthly_save_2020_path = monthly_path_2020 + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_monthly_save_2020_path):
    os.makedirs(nph_ws_10m_monthly_save_2020_path)

# 월간 저장 경로 설정 2021
nph_ta_monthly_save_2021_path = monthly_path_2021 + "\\nph_ta"
if not os.path.exists(nph_ta_monthly_save_2021_path):
    os.makedirs(nph_ta_monthly_save_2021_path)
nph_hm_monthly_save_2021_path = monthly_path_2021 + "\\nph_hm"
if not os.path.exists(nph_hm_monthly_save_2021_path):
    os.makedirs(nph_hm_monthly_save_2021_path)
nph_rn_60m_monthly_save_2021_path = monthly_path_2021 + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_monthly_save_2021_path):
    os.makedirs(nph_rn_60m_monthly_save_2021_path)
nph_ws_10m_monthly_save_2021_path = monthly_path_2021 + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_monthly_save_2021_path):
    os.makedirs(nph_ws_10m_monthly_save_2021_path)

# 월간 저장 경로 설정 2022
nph_ta_monthly_save_2022_path = monthly_path_2022 + "\\nph_ta"
if not os.path.exists(nph_ta_monthly_save_2022_path):
    os.makedirs(nph_ta_monthly_save_2022_path)
nph_hm_monthly_save_2022_path = monthly_path_2022 + "\\nph_hm"
if not os.path.exists(nph_hm_monthly_save_2022_path):
    os.makedirs(nph_hm_monthly_save_2022_path)
nph_rn_60m_monthly_save_2022_path = monthly_path_2022 + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_monthly_save_2022_path):
    os.makedirs(nph_rn_60m_monthly_save_2022_path)
nph_ws_10m_monthly_save_2022_path = monthly_path_2022 + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_monthly_save_2022_path):
    os.makedirs(nph_ws_10m_monthly_save_2022_path)


# 연간 저장 경로 설정
nph_ta_yearly_save_path = yearly_path + "\\nph_ta"
if not os.path.exists(nph_ta_yearly_save_path):
    os.makedirs(nph_ta_yearly_save_path)
nph_hm_yearly_save_path = yearly_path + "\\nph_hm"
if not os.path.exists(nph_hm_yearly_save_path):
    os.makedirs(nph_hm_yearly_save_path)
nph_rn_60m_yearly_save_path = yearly_path + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_yearly_save_path):
    os.makedirs(nph_rn_60m_yearly_save_path)
nph_ws_10m_yearly_save_path = yearly_path + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_yearly_save_path):
    os.makedirs(nph_ws_10m_yearly_save_path)


# 연간 이상치 제거 저장 경로 설정
nph_ta_yearly_rm_outlier_save_path = yearly_rm_outlier_path + "\\nph_ta"
if not os.path.exists(nph_ta_yearly_rm_outlier_save_path):
    os.makedirs(nph_ta_yearly_rm_outlier_save_path)
nph_hm_yearly_rm_outlier_save_path = yearly_rm_outlier_path + "\\nph_hm"
if not os.path.exists(nph_hm_yearly_rm_outlier_save_path):
    os.makedirs(nph_hm_yearly_rm_outlier_save_path)
nph_rn_60m_yearly_rm_outlier_save_path = yearly_rm_outlier_path + "\\nph_rn_60m"
if not os.path.exists(nph_rn_60m_yearly_rm_outlier_save_path):
    os.makedirs(nph_rn_60m_yearly_rm_outlier_save_path)
nph_ws_10m_yearly_rm_outlier_save_path = yearly_rm_outlier_path + "\\nph_ws_10m"
if not os.path.exists(nph_ws_10m_yearly_rm_outlier_save_path):
    os.makedirs(nph_ws_10m_yearly_rm_outlier_save_path)


# 2020년도 train_df의 electric_train.num 열의 유니크한 값 추출
unique_values_2020 = train_2020['electric_train.num'].unique()

# 2021년도 train_df의 electric_train.num 열의 유니크한 값 추출
unique_values_2021 = train_2021['electric_train.num'].unique()

# 2022년도 train_df의 electric_train.num 열의 유니크한 값 추출
unique_values_2022 = train_2022['electric_train.num'].unique()



def EDA_plot_elec_daily(num, feature, data, save_path):
    # data 에서 num번 격자의 데이터만 추출
    data_num = data[data['electric_train.num'] == num].copy()

    # 'electric_train.elec'가 음수인 경우 제거
    data_num = data_num[data_num['electric_train.elec'] >= 0]
    
    # 날짜 변수를 추가
    data_num.loc[:, 'date'] = data_num['electric_train.tm'].dt.date
    
    # 일 단위로 평균값 계산
    daily_data = data_num.groupby(['date']).mean().reset_index()
    
    # 시간 변수 제거 (이미 평균값으로 계산되었기 때문에 더 이상 필요 없음)
    daily_data = daily_data.drop(columns=['electric_train.tm'])
    
    # 필요한 열만 선택
    feature_col = f'electric_train.{feature}'
    daily_data = daily_data[[feature_col, 'electric_train.elec']]
    
    # 산점도 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature_col, y='electric_train.elec', data=daily_data)
    plt.title(f'Predited Power Demand vs. {feature} - {num} grid')
    plt.xlabel(feature)
    plt.ylabel('Predited Power Demand')
    plt.grid(True)
    plt.savefig(f"{save_path}\\{num}.png")
    plt.close()

    


def EDA_plot_elec_monthly(num, feature, data, save_path):
    # data 에서 num번 격자의 데이터만 추출
    data_num = data[data['electric_train.num'] == num].copy()

    # 'electric_train.elec'가 음수인 경우 제거
    data_num = data_num[data_num['electric_train.elec'] >= 0]
    
    # 월 변수를 추가
    data_num.loc[:, 'month'] = data_num['electric_train.tm'].dt.month
    
    # 월 단위로 평균값 계산
    monthly_data = data_num.groupby(['month']).mean().reset_index()
    
    # 시간 변수 제거 (이미 평균값으로 계산되었기 때문에 더 이상 필요 없음)
    monthly_data = monthly_data.drop(columns=['electric_train.tm'])
    
    # 필요한 열만 선택
    feature_col = f'electric_train.{feature}'
    monthly_data = monthly_data[[feature_col, 'electric_train.elec']]
    
    # 산점도 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature_col, y='electric_train.elec', data=monthly_data)
    plt.title(f'Predited Power Demand vs. {feature} - {num} grid')
    plt.xlabel(feature)
    plt.ylabel('Predited Power Demand')
    plt.grid(True)
    plt.savefig(f"{save_path}\\{num}.png")
    plt.close()


def remove_extreme_outliers(df, col, lower_bound, upper_bound):
    return df[(df[col] > lower_bound) & (df[col] < upper_bound)]


def EDA_plot_elec_yearly(num, feature, data, save_path):
    # num번 격자의 데이터만 추출
    data_num = data[data['electric_train.num'] == num].copy()
    
    # 필요한 열만 선택
    feature_col = f'electric_train.{feature}'
    data_num = data_num[[feature_col, 'electric_train.elec']]

    # 극단적인 이상치 제거 (임계값 설정: -1 < feature < 20)
    ## feature가 nph_rm_60m과 nph_ws_10m인 경우만 적용
    if feature == 'nph_rn_60m':
        data_num = remove_extreme_outliers(data_num, feature_col, -1, 30)
    elif feature == 'nph_ws_10m':
        data_num = remove_extreme_outliers(data_num, feature_col, -1, 20)
    
    # 'electric_train.elec'가 음수인 경우 제거
    data_num = data_num[data_num['electric_train.elec'] >= 0]

    # 산점도 시각화
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=feature_col, y='electric_train.elec', data = data_num)
    plt.title(f'Predited Power Demand vs. {feature} - {num} grid')
    plt.xlabel(feature)
    plt.ylabel('Predicted Power Demand')
    plt.grid(True)
    plt.savefig(f"{save_path}\\{num}_yearly.png")
    plt.close()





# 공통 격자 번호 추출
common_grids = np.intersect1d(train_2020['electric_train.num'].unique(), train_2021['electric_train.num'].unique())
common_grids = np.intersect1d(common_grids, train_2022['electric_train.num'].unique())

# 함수 호출 예시




# 공통 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 yearly 이상치 제거 시각화
idx = 0
for num in common_grids:
    start_time = time.time()
    EDA_plot_elec_yearly(num, 'nph_ta', train_df, nph_ta_yearly_rm_outlier_save_path)
    EDA_plot_elec_yearly(num, 'nph_hm', train_df, nph_hm_yearly_rm_outlier_save_path)
    EDA_plot_elec_yearly(num, 'nph_rn_60m', train_df, nph_rn_60m_yearly_rm_outlier_save_path)
    EDA_plot_elec_yearly(num, 'nph_ws_10m', train_df, nph_ws_10m_yearly_rm_outlier_save_path)
    end_time = time.time()
    idx += 1
    print(f"Elapsed time for {idx} EDA_plot_elec_yearly : {end_time - start_time} seconds")




# 모든 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 daily 시각화 - 2020
idx = 0
for num in unique_values_2020:
    start_time = time.time()
    EDA_plot_elec_daily(num, 'nph_ta', train_2020, nph_ta_daily_save_2020_path)
    EDA_plot_elec_daily(num, 'nph_hm', train_2020, nph_hm_daily_save_2020_path)
    EDA_plot_elec_daily(num, 'nph_rn_60m', train_2020, nph_rn_60m_daily_save_2020_path)
    EDA_plot_elec_daily(num, 'nph_ws_10m', train_2020, nph_ws_10m_daily_save_2020_path)
    end_time = time.time()
    idx += 1
    print(f"Elapsed time for {idx} EDA_plot_elec_daily : {end_time - start_time} seconds")


# 모든 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 daily 시각화 - 2021
idx = 0
for num in unique_values_2021:
    start_time = time.time()
    EDA_plot_elec_daily(num, 'nph_ta', train_2021, nph_ta_daily_save_2021_path)
    EDA_plot_elec_daily(num, 'nph_hm', train_2021, nph_hm_daily_save_2021_path)
    EDA_plot_elec_daily(num, 'nph_rn_60m', train_2021, nph_rn_60m_daily_save_2021_path)
    EDA_plot_elec_daily(num, 'nph_ws_10m', train_2021, nph_ws_10m_daily_save_2021_path)
    end_time = time.time()
    idx += 1
    print(f"Elapsed time for {idx} EDA_plot_elec_daily : {end_time - start_time} seconds")

# 모든 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 daily 시각화 - 2022
idx = 0
for num in unique_values_2022:
    start_time = time.time()
    EDA_plot_elec_daily(num, 'nph_ta', train_2022, nph_ta_daily_save_2022_path)
    EDA_plot_elec_daily(num, 'nph_hm', train_2022, nph_hm_daily_save_2022_path)
    EDA_plot_elec_daily(num, 'nph_rn_60m', train_2022, nph_rn_60m_daily_save_2022_path)
    EDA_plot_elec_daily(num, 'nph_ws_10m', train_2022, nph_ws_10m_daily_save_2022_path)
    end_time = time.time()
    idx += 1
    print(f"Elapsed time for {idx} EDA_plot_elec_daily : {end_time - start_time} seconds")


# 모든 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 monthly 시각화 - 2020
idx = 0
for num in unique_values_2020:
    start_time = time.time()
    EDA_plot_elec_monthly(num, 'nph_ta', train_2020, nph_ta_monthly_save_2020_path)
    EDA_plot_elec_monthly(num, 'nph_hm', train_2020, nph_hm_monthly_save_2020_path)
    EDA_plot_elec_monthly(num, 'nph_rn_60m', train_2020, nph_rn_60m_monthly_save_2020_path)
    EDA_plot_elec_monthly(num, 'nph_ws_10m', train_2020, nph_ws_10m_monthly_save_2020_path)
    end_time = time.time()
    idx += 1
    print(f"Elapsed time for {idx} EDA_plot_elec_monthly : {end_time - start_time} seconds")


# 모든 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 monthly 시각화 - 2021
idx = 0
for num in unique_values_2021:
    start_time = time.time()
    EDA_plot_elec_monthly(num, 'nph_ta', train_2021, nph_ta_monthly_save_2021_path)
    EDA_plot_elec_monthly(num, 'nph_hm', train_2021, nph_hm_monthly_save_2021_path)
    EDA_plot_elec_monthly(num, 'nph_rn_60m', train_2021, nph_rn_60m_monthly_save_2021_path)
    EDA_plot_elec_monthly(num, 'nph_ws_10m', train_2021, nph_ws_10m_monthly_save_2021_path)
    end_time = time.time()
    idx += 1
    print(f"Elapsed time for {idx} EDA_plot_elec_monthly : {end_time - start_time} seconds")


# 모든 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 monthly 시각화 - 2022
idx = 0
for num in unique_values_2022:
    start_time = time.time()
    EDA_plot_elec_monthly(num, 'nph_ta', train_2022, nph_ta_monthly_save_2022_path)
    EDA_plot_elec_monthly(num, 'nph_hm', train_2022, nph_hm_monthly_save_2022_path)
    EDA_plot_elec_monthly(num, 'nph_rn_60m', train_2022, nph_rn_60m_monthly_save_2022_path)
    EDA_plot_elec_monthly(num, 'nph_ws_10m', train_2022, nph_ws_10m_monthly_save_2022_path)
    end_time = time.time()
    idx += 1
    print(f"Elapsed time for {idx} EDA_plot_elec_monthly : {end_time - start_time} seconds")


# # 공통 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 yearly 시각화
# idx = 0
# for num in common_grids:
#     start_time = time.time()
#     EDA_plot_elec_yearly(num, 'nph_ta', train_df, nph_ta_yearly_save_path)
#     EDA_plot_elec_yearly(num, 'nph_hm', train_df, nph_hm_yearly_save_path)
#     EDA_plot_elec_yearly(num, 'nph_rn_60m', train_df, nph_rn_60m_yearly_save_path)
#     EDA_plot_elec_yearly(num, 'nph_ws_10m', train_df, nph_ws_10m_yearly_save_path)
#     end_time = time.time()
#     idx += 1
#     print(f"Elapsed time for {idx} EDA_plot_elec_yearly : {end_time - start_time} seconds")


