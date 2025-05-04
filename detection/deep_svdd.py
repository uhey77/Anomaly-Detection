import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

class DeepSVDDDetector:
    """Deep Support Vector Data Description による異常検知"""
    
    def __init__(self, threshold=0.9, epochs=50, batch_size=32, random_state=42):
        """
        Deep SVDD 検出器を初期化
        
        Args:
            threshold (float): 異常と判定する確率の閾値（0-1）
            epochs (int): トレーニングのエポック数
            batch_size (int): バッチサイズ
            random_state (int): 乱数シード
        """
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.center = None
        
    def _create_model(self, input_dim):
        """
        Deep SVDD モデルを作成
        
        Args:
            input_dim (int): 入力次元数
            
        Returns:
            Model: Kerasモデル
        """
        tf.random.set_seed(self.random_state)
        
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(2, activation=None)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _svdd_loss(self, y_true, y_pred):
        """
        Deep SVDD の損失関数
        
        Args:
            y_true: 真のラベル（使用しない）
            y_pred: モデル出力
            
        Returns:
            float: 損失値
        """
        # 中心からの距離の二乗の平均
        return tf.reduce_mean(tf.square(y_pred - self.center))
    
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
        
        # モデルの中心を初期化
        self.center = tf.zeros([2])
        
        # モデルを作成
        self.model = self._create_model(X_scaled.shape[1])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=self._svdd_loss)
        
        # データの95%を正常と仮定してトレーニング
        self.model.fit(
            X_scaled, np.zeros((X_scaled.shape[0], 2)),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        # 異常スコアを計算
        representations = self.model.predict(X_scaled)
        distances = np.sum(np.square(representations - self.center.numpy()), axis=1)
        
        # 距離の分布を調べて異常を検出
        distance_threshold = np.percentile(distances, self.threshold * 100)
        
        # 結果をデータフレームに追加
        df_features['anomaly_score'] = distances
        df_features['is_anomaly'] = distances > distance_threshold
        
        # 変化率を追加
        df_features['pct_change'] = df_features[value_column].pct_change() * 100
        
        # 異常のみを抽出
        anomalies = df_features[df_features['is_anomaly']].copy()
        
        return anomalies