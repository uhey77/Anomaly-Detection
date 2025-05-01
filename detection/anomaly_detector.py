import numpy as np
import pandas as pd
from scipy import stats


class AnomalyDetector:
    """金融時系列データ用の基本的な異常検知器"""
    
    def __init__(self, method='z_score', threshold=3.0):
        """
        異常検知器を初期化
        
        Args:
            method (str): 検出方法 ('z_score', 'iqr', または 'moving_avg')
            threshold (float): 異常検知の閾値
        """
        self.method = method
        self.threshold = threshold
        
    def detect(self, data, date_column='Date', value_column='Close'):
        """
        時系列データ内の異常を検出
        
        Args:
            data (pd.DataFrame): 入力時系列データ
            date_column (str): 日付カラム名
            value_column (str): 値カラム名
            
        Returns:
            pd.DataFrame: 異常フラグ付きのデータ
        """
        df = data.copy()
        
        if self.method == 'z_score':
            # Z-scoreメソッド
            z_scores = stats.zscore(df[value_column])
            df['z_score'] = z_scores
            df['is_anomaly'] = abs(z_scores) > self.threshold
            
        elif self.method == 'iqr':
            # IQRメソッド
            Q1 = df[value_column].quantile(0.25)
            Q3 = df[value_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            df['is_anomaly'] = (df[value_column] < lower_bound) | (df[value_column] > upper_bound)
            
        elif self.method == 'moving_avg':
            # 移動平均メソッド
            window = 20  # 20日ウィンドウ
            df['moving_avg'] = df[value_column].rolling(window=window).mean()
            df['moving_std'] = df[value_column].rolling(window=window).std()
            
            # 最初の'window'日をスキップ
            df['is_anomaly'] = False
            df.loc[window:, 'is_anomaly'] = abs(df[value_column] - df['moving_avg']) > self.threshold * df['moving_std']
        
        # 日次変化率を計算
        df['pct_change'] = df[value_column].pct_change() * 100
        
        # 重要な変化率を持つ異常を見つける
        df['is_significant_change'] = abs(df['pct_change']) > self.threshold * df['pct_change'].std()
        
        # 最終的な異常は、メソッド固有の異常または重要な変化率のいずれか
        df['is_anomaly'] = df['is_anomaly'] | df['is_significant_change']
        
        # 異常のみを取得
        anomalies = df[df['is_anomaly']].copy()
        
        return anomalies
