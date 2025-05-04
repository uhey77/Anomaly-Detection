import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TimeSeriesModel:
    """時系列予測モデルの基底クラス"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, df, target_col='Close', lookback=60):
        """
        モデル用のデータを準備
        
        Args:
            df (pd.DataFrame): 入力データ
            target_col (str): 目標変数のカラム名
            lookback (int): 遡る期間
            
        Returns:
            tuple: (X_train, y_train)
        """
        data = df[target_col].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X).reshape(len(X), lookback, 1), np.array(y)
    
    def predict(self, X):
        """
        予測を行う
        
        Args:
            X: 入力データ
            
        Returns:
            np.array: 予測値
        """
        if self.model is None:
            raise ValueError("モデルがトレーニングされていません")
        
        return self.model.predict(X)
    
    def inverse_transform(self, scaled_data):
        """
        スケールされたデータを元のスケールに戻す
        
        Args:
            scaled_data: スケールされたデータ
            
        Returns:
            np.array: 元のスケールのデータ
        """
        return self.scaler.inverse_transform(scaled_data)


class LSTMModel(TimeSeriesModel):
    """LSTM時系列予測モデル"""
    
    def __init__(self, units=50, dropout=0.2):
        """
        LSTMモデルを初期化
        
        Args:
            units (int): LSTMユニット数
            dropout (float): ドロップアウト率
        """
        super().__init__()
        self.units = units
        self.dropout = dropout
        
    def build_model(self, lookback=60):
        """
        LSTMモデルを構築
        
        Args:
            lookback (int): 遡る期間
        """
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(lookback, 1)))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        
    def train(self, df, target_col='Close', lookback=60, epochs=50, batch_size=32, validation_split=0.2):
        """
        モデルをトレーニング
        
        Args:
            df (pd.DataFrame): 入力データ
            target_col (str): 目標変数のカラム名
            lookback (int): 遡る期間
            epochs (int): エポック数
            batch_size (int): バッチサイズ
            validation_split (float): 検証データの割合
            
        Returns:
            history: トレーニング履歴
        """
        X_train, y_train = self.prepare_data(df, target_col, lookback)
        
        if self.model is None:
            self.build_model(lookback)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def forecast(self, df, days_ahead=30, target_col='Close', lookback=60):
        """
        将来の価格を予測
        
        Args:
            df (pd.DataFrame): 入力データ
            days_ahead (int): 予測日数
            target_col (str): 目標変数のカラム名
            lookback (int): 遡る期間
            
        Returns:
            pd.DataFrame: 予測結果
        """
        if self.model is None:
            raise ValueError("モデルがトレーニングされていません")
        
        # 最後のlookback日間のデータを取得
        last_data = df[target_col].values[-lookback:].reshape(-1, 1)
        scaled_last_data = self.scaler.transform(last_data)
        
        # 予測用の入力データを準備
        X_pred = scaled_last_data.reshape(1, lookback, 1)
        
        # 予測結果を格納するリスト
        predictions = []
        
        # days_ahead日分の予測
        for _ in range(days_ahead):
            # 予測
            next_pred = self.model.predict(X_pred)
            predictions.append(next_pred[0, 0])
            
            # 次の入力データを更新
            X_pred = np.append(X_pred[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
        
        # 予測結果を元のスケールに戻す
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # 予測日付を生成
        last_date = df['Date'].iloc[-1]
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        
        # 予測結果をデータフレームに変換
        forecast_df = pd.DataFrame({
            'Date': dates,
            'predicted_price': predictions.flatten()
        })
        
        return forecast_df


class TimesFMModel(TimeSeriesModel):
    """TimesFM時系列予測モデル（簡略化したTransformerベースモデル）"""
    
    def __init__(self, d_model=64, num_heads=4, num_layers=2, dropout=0.1):
        """
        TimesFMモデルを初期化
        
        Args:
            d_model (int): モデルの次元
            num_heads (int): Attention Headsの数
            num_layers (int): Transformer Layersの数
            dropout (float): ドロップアウト率
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
    def build_model(self, lookback=60):
        """
        TimesFMモデルを構築（Transformerベース）
        
        Args:
            lookback (int): 遡る期間
        """
        # 入力層
        inputs = tf.keras.Input(shape=(lookback, 1))
        
        # 入力を埋め込み次元に変換
        x = tf.keras.layers.Conv1D(filters=self.d_model, kernel_size=1, activation='relu')(inputs)
        
        # Transformer Encoder Layers
        for _ in range(self.num_layers):
            # Multi-Head Self Attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.d_model // self.num_heads
            )(x, x)
            
            # Add & Norm
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed Forward Network
            ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(self.d_model * 4, activation='relu'),
                tf.keras.layers.Dense(self.d_model),
                tf.keras.layers.Dropout(self.dropout)
            ])
            
            # Add & Norm
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))
        
        # グローバルプーリングで時系列データを集約
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # 出力層
        outputs = tf.keras.layers.Dense(1)(x)
        
        # モデルの構築
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        
    def train(self, df, target_col='Close', lookback=60, epochs=50, batch_size=32, validation_split=0.2):
        """
        モデルをトレーニング
        
        Args:
            df (pd.DataFrame): 入力データ
            target_col (str): 目標変数のカラム名
            lookback (int): 遡る期間
            epochs (int): エポック数
            batch_size (int): バッチサイズ
            validation_split (float): 検証データの割合
            
        Returns:
            history: トレーニング履歴
        """
        X_train, y_train = self.prepare_data(df, target_col, lookback)
        
        if self.model is None:
            self.build_model(lookback)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def forecast(self, df, days_ahead=30, target_col='Close', lookback=60):
        """
        将来の価格を予測
        
        Args:
            df (pd.DataFrame): 入力データ
            days_ahead (int): 予測日数
            target_col (str): 目標変数のカラム名
            lookback (int): 遡る期間
            
        Returns:
            pd.DataFrame: 予測結果
        """
        if self.model is None:
            raise ValueError("モデルがトレーニングされていません")
        
        # 最後のlookback日間のデータを取得
        last_data = df[target_col].values[-lookback:].reshape(-1, 1)
        scaled_last_data = self.scaler.transform(last_data)
        
        # 予測用の入力データを準備
        X_pred = scaled_last_data.reshape(1, lookback, 1)
        
        # 予測結果を格納するリスト
        predictions = []
        
        # days_ahead日分の予測
        for _ in range(days_ahead):
            # 予測
            next_pred = self.model.predict(X_pred)
            predictions.append(next_pred[0, 0])
            
            # 次の入力データを更新
            X_pred = np.append(X_pred[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
        
        # 予測結果を元のスケールに戻す
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        # 予測日付を生成
        last_date = df['Date'].iloc[-1]
        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        
        # 予測結果をデータフレームに変換
        forecast_df = pd.DataFrame({
            'Date': dates,
            'predicted_price': predictions.flatten()
        })
        
        return forecast_df
