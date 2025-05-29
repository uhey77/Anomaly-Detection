import pandas as pd
import numpy as np

class SignalGenerator:
    """異常検知結果から売買シグナルを生成するクラス"""
    
    def __init__(self, threshold=0.5, window_size=5):
        """
        シグナル生成器を初期化
        
        Args:
            threshold (float): シグナル生成の閾値
            window_size (int): 価格トレンド判定の窓サイズ
        """
        self.threshold = threshold
        self.window_size = window_size
        
    def generate_signals(self, df, anomalies):
        """
        異常検知結果から売買シグナルを生成
        
        Args:
            df (pd.DataFrame): 元の時系列データ
            anomalies (pd.DataFrame): 検出された異常
            
        Returns:
            pd.DataFrame: 売買シグナル付きのデータフレーム
        """
        # 元のデータをコピー
        signals_df = df.copy()
        
        # 初期シグナルを0（ホールド）に設定
        signals_df['signal'] = 0
        
        # 異常強度カラムを追加
        signals_df['anomaly_score'] = 0.0
        
        # pct_changeが含まれていない場合は計算して追加
        if 'pct_change' not in signals_df.columns:
            signals_df['pct_change'] = signals_df['Close'].pct_change() * 100
        
        if not anomalies.empty:
            # 異常データの方向性を判定
            for idx, anomaly in anomalies.iterrows():
                date = anomaly['Date']
                
                # データフレーム内のこの日付のインデックスを取得
                try:
                    df_idx = signals_df[signals_df['Date'] == date].index[0]
                    
                    # 方向性の判定（過去数日間の傾向に基づく）
                    if df_idx >= self.window_size:
                        # 過去window_size日間の平均変化率
                        past_trend = signals_df.loc[df_idx-self.window_size:df_idx-1, 'Close'].pct_change().mean()
                        
                        # 当日の変化率
                        current_change = anomaly['pct_change'] / 100  # パーセントから小数に変換
                        
                        # 異常スコアの計算（Z-scoreの絶対値または指定された異常スコア）
                        if 'z_score' in anomaly:
                            anomaly_score = abs(anomaly['z_score'])
                        else:
                            # 異常の強さを変化率の大きさで代用
                            anomaly_score = abs(current_change) / 0.01  # 1%の変化を基準に正規化
                        
                        # 方向性判断
                        if current_change > 0 and (current_change > past_trend * 1.5):
                            # 上昇異常（買いシグナル）
                            signals_df.loc[df_idx, 'signal'] = 1
                        elif current_change < 0 and (current_change < past_trend * 1.5):
                            # 下降異常（売りシグナル）
                            signals_df.loc[df_idx, 'signal'] = -1
                        
                        # 異常スコアを保存 - float32との互換性問題を回避するために明示的に型変換
                        signals_df.loc[df_idx, 'anomaly_score'] = float(anomaly_score)
                        
                except IndexError:
                    continue
        
        return signals_df
    
    def forecast_adjustment(self, signals_df, forecast_df, weight=0.5):
        """
        シグナルに基づいて予測価格を補正
        
        Args:
            signals_df (pd.DataFrame): シグナル付きデータフレーム
            forecast_df (pd.DataFrame): 予測データフレーム
            weight (float): 異常に基づく補正の重み
            
        Returns:
            pd.DataFrame: 補正された予測
        """
        adjusted_forecast = forecast_df.copy()
        
        # シグナルのある日付のみを抽出
        signal_dates = signals_df[signals_df['signal'] != 0]
        
        if not signal_dates.empty:
            for idx, row in signal_dates.iterrows():
                date = row['Date']
                signal = row['signal']
                score = row['anomaly_score']
                
                # この日付以降の予測を見つける
                future_idx = adjusted_forecast[adjusted_forecast['Date'] >= date].index
                
                if len(future_idx) > 0:
                    # 補正係数の計算（シグナルの方向と強度に基づく）
                    # 徐々に減衰する影響を計算
                    for i, future_i in enumerate(future_idx):
                        decay = np.exp(-0.1 * i)  # 指数関数的減衰
                        adjustment = signal * score * weight * decay
                        
                        # 予測価格の補正（現在の予測値に対する割合で）
                        current_pred = adjusted_forecast.loc[future_i, 'predicted_price']
                        adjusted_forecast.loc[future_i, 'predicted_price'] = current_pred * (1 + adjustment/100)
        
        return adjusted_forecast
