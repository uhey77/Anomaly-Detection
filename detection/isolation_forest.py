import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationForestDetector:
    """Isolation Forest による異常検知"""
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        """
        Isolation Forest 検出器を初期化
        
        Args:
            contamination (float): データにおける異常の割合の期待値
            n_estimators (int): 決定木の数
            random_state (int): 乱数シード
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def detect(self, data, date_column='Date', value_column='Close', extra_features=None):
        """
        時系列データ内の異常を検出
        
        Args:
            data (pd.DataFrame): 入力時系列データ
            date_column (str): 日付カラム名
            value_column (str): 値カラム名
            extra_features (list): 追加の特徴量カラム名のリスト
            
        Returns:
            pd.DataFrame: 異常フラグ付きのデータ
        """
        df = data.copy()
        
        # 基本的な特徴量を作成
        df['returns'] = df[value_column].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        df['rolling_mean_5'] = df[value_column].rolling(window=5).mean()
        df['rolling_std_5'] = df[value_column].rolling(window=5).std()
        df['rolling_z'] = (df[value_column] - df['rolling_mean_5']) / df['rolling_std_5']
        
        # 移動平均乖離率を計算
        df['ma_20'] = df[value_column].rolling(window=20).mean()
        df['ma_50'] = df[value_column].rolling(window=50).mean()
        df['ma_ratio_20'] = df[value_column] / df['ma_20']
        df['ma_ratio_50'] = df[value_column] / df['ma_50']
        
        # 特徴量リストを作成
        features = [
            'returns', 'log_returns', 'rolling_z',
            'ma_ratio_20', 'ma_ratio_50'
        ]
        
        # 追加の特徴量があれば追加
        if extra_features:
            features.extend(extra_features)
        
        # 初期データ処理中の欠損値を削除
        df_features = df.dropna()
        
        # 特徴量を抽出
        X = df_features[features].values
        
        # 特徴量をスケーリング
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forestモデルを作成・適用
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # モデルを適合
        self.model.fit(X_scaled)
        
        # 異常スコアを予測
        anomaly_scores = self.model.score_samples(X_scaled)
        anomaly_labels = self.model.predict(X_scaled)
        
        # 結果をデータフレームに追加
        df_features['anomaly_score'] = anomaly_scores
        df_features['is_anomaly'] = anomaly_labels == -1  # -1が異常、1が正常
        
        # 変化率を追加
        df_features['pct_change'] = df_features[value_column].pct_change() * 100
        
        # 異常のみを抽出
        anomalies = df_features[df_features['is_anomaly']].copy()
        
        return anomalies