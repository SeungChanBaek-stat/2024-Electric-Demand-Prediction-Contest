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


# 이상치 제거 함수
def remove_extreme_outliers(df, col, lower_bound, upper_bound):
    return df[(df[col] > lower_bound) & (df[col] < upper_bound)]

# 이상치 제거 및 음수 값 제거
def clean_data(data):
    cleaned_data = data.copy()
    for feature in ['nph_ta', 'nph_hm', 'nph_rn_60m', 'nph_ws_10m']:
        feature_col = f'electric_train.{feature}'
        if feature == 'nph_rn_60m':
            cleaned_data = remove_extreme_outliers(cleaned_data, feature_col, -1, 30)
        elif feature == 'nph_ws_10m':
            cleaned_data = remove_extreme_outliers(cleaned_data, feature_col, -1, 20)
        cleaned_data = cleaned_data[cleaned_data['electric_train.elec'] >= 0]
    return cleaned_data


# 데이터 클리닝
cleaned_train_df = clean_data(train_df)
print(f'Length of cleaned_train_df: {len(cleaned_train_df)}')

# 이상치가 제거된 데이터셋 저장
cleaned_train_df.to_csv(os.path.join(data_dir, 'electric_train_cleaned.csv'), index=False)

# electric_train.tm 변수를 시계열로 변환
cleaned_train_df['electric_train.tm'] = pd.to_datetime(cleaned_train_df['electric_train.tm'])

# 2020, 2021, 2022 데이터 분리
cleaned_train_df['year'] = cleaned_train_df['electric_train.tm'].dt.year
train_2020 = cleaned_train_df[cleaned_train_df['year'] == 2020]
train_2021 = cleaned_train_df[cleaned_train_df['year'] == 2021]
train_2022 = cleaned_train_df[cleaned_train_df['year'] == 2022]


# 각 연도의 유니크한 격자 번호 추출
unique_values_2020 = train_2020['electric_train.num'].unique()
unique_values_2021 = train_2021['electric_train.num'].unique()
unique_values_2022 = train_2022['electric_train.num'].unique()

# 공통 격자 번호 추출
common_grids = np.intersect1d(unique_values_2020, unique_values_2021)
common_grids = np.intersect1d(common_grids, unique_values_2022)




# 시각화 함수
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
    plt.title(f'Predicted Power Demand vs. {feature} - {num} grid')
    plt.xlabel(feature)
    plt.ylabel('Predicted Power Demand')
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
    plt.title(f'Predicted Power Demand vs. {feature} - {num} grid')
    plt.xlabel(feature)
    plt.ylabel('Predicted Power Demand')
    plt.grid(True)
    plt.savefig(f"{save_path}\\{num}.png")
    plt.close()



def EDA_plot_elec_yearly(num, feature, data, save_path):
    # num번 격자의 데이터만 추출
    data_num = data[data['electric_train.num'] == num].copy()
    
    # 필요한 열만 선택
    feature_col = f'electric_train.{feature}'
    data_num = data_num[[feature_col, 'electric_train.elec']]

    # 극단적인 이상치 제거 (임계값 설정: -1 < feature < 20)
    ## feature가 nph_rn_60m과 nph_ws_10m인 경우만 적용
    if feature == 'nph_rn_60m':
        data_num = remove_extreme_outliers(data_num, feature_col, -1, 30)
    elif feature == 'nph_ws_10m':
        data_num = remove_extreme_outliers(data_num, feature_col, -1, 20)
    
    # 'electric_train.elec'가 음수인 경우 제거
    data_num = data_num[data_num['electric_train.elec'] >= 0]

    # 산점도 시각화
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x=feature_col, y='electric_train.elec', data=data_num)
    plt.title(f'Predicted Power Demand vs. {feature} - {num} grid')
    plt.xlabel(feature)
    plt.ylabel('Predicted Power Demand')
    plt.grid(True)
    plt.savefig(f"{save_path}\\{num}_yearly.png")
    plt.close()











# 저장 경로 설정
save_dir = "C:\\Users\\AAA\\2024-Electric-Demand-Prediction-Contest\\eda\\cleaned"
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






# 공통 격자에 대해서, nph_ta, nph_hm, nph_rn_60m, nph_ws_10m 변수 yearly 시각화
idx = 0
for num in common_grids:
    start_time = time.time()
    EDA_plot_elec_yearly(num, 'nph_ta', cleaned_train_df, nph_ta_yearly_save_path)
    EDA_plot_elec_yearly(num, 'nph_hm', cleaned_train_df, nph_hm_yearly_save_path)
    EDA_plot_elec_yearly(num, 'nph_rn_60m', cleaned_train_df, nph_rn_60m_yearly_save_path)
    EDA_plot_elec_yearly(num, 'nph_ws_10m', cleaned_train_df, nph_ws_10m_yearly_save_path)
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