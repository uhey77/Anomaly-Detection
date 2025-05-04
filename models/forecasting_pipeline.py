import pandas as pd
import numpy as np
from utils.signal_generator import SignalGenerator
from models.time_series_models import LSTMModel, TimesFMModel

class ForecastingPipeline:
    """異常検知から予測までのパイプライン"""
    
    def __init__(self, forecasting_model='lstm', signal_threshold=0.5, window_size=5, lookback=60):
        """
        予測パイプラインを初期化
        
        Args:
            forecasting_model (str): 使用する予測モデル ('lstm' または 'timesfm')
            signal_threshold (float): シグナル生成の閾値
            window_size (int): 価格トレンド判定の窓サイズ
            lookback (int): 予測モデルの遡る期間
        """
        self.forecasting_model_type = forecasting_model
        self.signal_threshold = signal_threshold
        self.window_size = window_size
        self.lookback = lookback
        
        # シグナル生成器
        self.signal_generator = SignalGenerator(threshold=signal_threshold, window_size=window_size)
        
        # 予測モデル
        if forecasting_model.lower() == 'lstm':
            self.forecasting_model = LSTMModel()
        elif forecasting_model.lower() == 'timesfm':
            self.forecasting_model = TimesFMModel()
        else:
            raise ValueError(f"未対応の予測モデル: {forecasting_model}")
    
    def process(self, df, anomalies, train_model=True, days_ahead=30, adjustment_weight=0.5):
        """
        異常検知から予測までの処理を実行
        
        Args:
            df (pd.DataFrame): 元の時系列データ
            anomalies (pd.DataFrame): 検出された異常
            train_model (bool): モデルを再トレーニングするかどうか
            days_ahead (int): 予測日数
            adjustment_weight (float): 異常に基づく補正の重み
            
        Returns:
            tuple: (signals_df, forecast_df, adjusted_forecast_df)
        """
        # 1. 異常から売買シグナルを生成
        signals_df = self.signal_generator.generate_signals(df, anomalies)
        
        # 2. 予測モデルをトレーニング（必要な場合）
        if train_model:
            self.forecasting_model.train(df, lookback=self.lookback)
        
        # 3. 将来の価格を予測
        forecast_df = self.forecasting_model.forecast(df, days_ahead=days_ahead, lookback=self.lookback)
        
        # 4. シグナルに基づいて予測価格を補正
        adjusted_forecast_df = self.signal_generator.forecast_adjustment(
            signals_df, forecast_df, weight=adjustment_weight
        )
        
        return signals_df, forecast_df, adjusted_forecast_df
