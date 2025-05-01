import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_sample_data(start_date="1980-01-01", end_date="2023-12-31"):
    """
    サンプルのS&P 500データを生成
    
    Args:
        start_date (str): 開始日
        end_date (str): 終了日
        
    Returns:
        pd.DataFrame: サンプルデータ
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    np.random.seed(42)
    
    # 基本的な株価動き（上昇トレンド + ランダムノイズ）
    n = len(dates)
    base_value = 100
    trend = np.linspace(0, 2, n)
    noise = np.random.normal(0, 0.02, n)
    returns = 0.0005 + trend * 0.0001 + noise
    
    # 累積リターンを計算
    cum_returns = np.cumprod(1 + returns)
    values = base_value * cum_returns
    
    # 特定の日に異常を導入
    # 1987年10月19日（ブラックマンデー）
    black_monday = pd.Timestamp('1987-10-19')
    if black_monday in dates:
        idx = dates.get_loc(black_monday)
        values[idx] = values[idx-1] * 0.774  # 22.6%の下落
    
    # 2008年10月13日の急騰
    oct_2008_surge = pd.Timestamp('2008-10-13')
    if oct_2008_surge in dates:
        idx = dates.get_loc(oct_2008_surge)
        values[idx] = values[idx-1] * 1.118  # 11.8%の上昇
    
    # 2020年3月16日のCOVID-19による下落
    covid_crash = pd.Timestamp('2020-03-16')
    if covid_crash in dates:
        idx = dates.get_loc(covid_crash)
        values[idx] = values[idx-1] * 0.88  # 12%の下落
    
    # データフレームを作成
    df = pd.DataFrame({
        'Date': dates,
        'Close': values
    })
    
    return df

def save_sample_data(df, filename="sp500_sample.csv"):
    """
    サンプルデータをCSVファイルに保存
    
    Args:
        df (pd.DataFrame): 保存するデータフレーム
        filename (str): 保存先のファイル名
    """
    # データディレクトリを確認
    os.makedirs("data", exist_ok=True)
    
    # CSVとして保存
    df.to_csv(f"data/{filename}", index=False)
    print(f"サンプルデータを data/{filename} に保存しました")

def plot_anomalies(data, anomalies, output_file=None):
    """
    時系列データと検出された異常をプロット
    
    Args:
        data (pd.DataFrame): 元の時系列データ
        anomalies (pd.DataFrame): 検出された異常
        output_file (str, optional): 出力ファイル名
    """
    plt.figure(figsize=(12, 6))
    
    # 元のデータをプロット
    plt.plot(data['Date'], data['Close'], label='S&P 500', color='blue')
    
    # 異常をプロット
    if not anomalies.empty:
        plt.scatter(anomalies['Date'], anomalies['Close'], color='red', s=50, label='検出された異常')
    
    plt.title('S&P 500と検出された異常')
    plt.xlabel('日付')
    plt.ylabel('価格')
    plt.legend()
    plt.grid(True)
    
    # x軸の日付フォーマットを調整
    plt.gcf().autofmt_xdate()
    
    # 出力ファイルが指定されている場合、画像を保存
    if output_file:
        # 結果ディレクトリを確認
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{output_file}")
        print(f"プロットを results/{output_file} に保存しました")
    else:
        plt.show()
