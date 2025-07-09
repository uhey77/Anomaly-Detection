import os
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# 設定ファイルをインポート
from config import *

# 異常検知器をインポート
from detection.anomaly_detector import AnomalyDetector

# ユーティリティ関数をインポート
from utils.data_utils import load_sample_data, save_sample_data, load_multi_indicator_data

# LLMクライアントをインポート
from utils.llm_clients import OpenAIClient, HuggingFaceClient, MockLLMClient

# エージェントをインポート
from agents.web_agent import WebInformationAgent
from agents.knowledge_agent import KnowledgeBaseAgent
from agents.crosscheck_agent import CrossCheckAgent
from agents.report_agent import ReportIntegrationAgent
from agents.manager_agent import ManagerAgent

# 評価クラスをインポート
from evaluation import AnomalyEvaluator

# シグナル生成器と予測モデルをインポート
from utils.signal_generator import SignalGenerator
from models.time_series_models import LSTMModel, TimesFMModel
from models.forecasting_pipeline import ForecastingPipeline

# リアルタイムデータプロバイダーをインポート
try:
    from realtime_data_provider import RealTimeDataProvider
    REALTIME_AVAILABLE = True
    print("✅ リアルタイムデータプロバイダーが利用可能です")
except ImportError:
    REALTIME_AVAILABLE = False
    print("⚠️ リアルタイムデータプロバイダーが見つかりません")

# DataManagerをインポート
try:
    from utils.data_utils import DataManager
    data_manager = DataManager()
    rt_provider = RealTimeDataProvider() if REALTIME_AVAILABLE else None
    print("✅ データマネージャーを初期化しました")
except ImportError:
    print("⚠️ DataManagerのインポートに失敗しました")
    data_manager = None
    rt_provider = None


# 修正版SignalGeneratorクラス（データ型問題対応）
class FixedSignalGenerator:
    """異常検知結果から売買シグナルを生成するクラス（データ型問題修正版）"""
    
    def __init__(self, threshold=0.5, window_size=5):
        self.threshold = threshold
        self.window_size = window_size
        
    def generate_signals(self, df, anomalies):
        """異常検知結果から売買シグナルを生成"""
        signals_df = df.copy()
        
        # データ型を統一（float64に）
        if 'Close' in signals_df.columns:
            signals_df['Close'] = signals_df['Close'].astype(np.float64)
        
        signals_df['signal'] = 0
        signals_df['anomaly_score'] = np.float64(0.0)
        
        if 'pct_change' not in signals_df.columns:
            signals_df['pct_change'] = signals_df['Close'].pct_change() * 100
        
        signals_df['pct_change'] = signals_df['pct_change'].astype(np.float64)
        
        if not anomalies.empty:
            for idx, anomaly in anomalies.iterrows():
                date = anomaly['Date']
                
                try:
                    matching_rows = signals_df[signals_df['Date'] == date]
                    if matching_rows.empty:
                        continue
                    
                    df_idx = matching_rows.index[0]
                    
                    if df_idx >= self.window_size:
                        past_data = signals_df.loc[df_idx-self.window_size:df_idx-1, 'Close']
                        if len(past_data) > 1:
                            past_trend = past_data.pct_change().mean()
                        else:
                            past_trend = 0.0
                        
                        if 'pct_change' in anomaly:
                            current_change = float(anomaly['pct_change']) / 100
                        else:
                            if df_idx > 0:
                                prev_price = float(signals_df.loc[df_idx-1, 'Close'])
                                curr_price = float(anomaly['Close'])
                                current_change = (curr_price - prev_price) / prev_price
                            else:
                                current_change = 0.0
                        
                        if 'z_score' in anomaly:
                            anomaly_score = abs(float(anomaly['z_score']))
                        elif 'anomaly_score' in anomaly:
                            anomaly_score = abs(float(anomaly['anomaly_score']))
                        else:
                            anomaly_score = abs(current_change) / 0.01
                        
                        trend_threshold = 1.5
                        if current_change > 0 and (current_change > past_trend * trend_threshold):
                            signals_df.loc[df_idx, 'signal'] = 1
                        elif current_change < 0 and (current_change < past_trend * trend_threshold):
                            signals_df.loc[df_idx, 'signal'] = -1
                        
                        signals_df.loc[df_idx, 'anomaly_score'] = np.float64(anomaly_score)
                        
                except (IndexError, KeyError, ValueError, TypeError) as e:
                    print(f"シグナル生成エラー（日付: {date}）: {e}")
                    continue
        
        return signals_df
    
    def forecast_adjustment(self, signals_df, forecast_df, weight=0.5):
        """シグナルに基づいて予測価格を補正"""
        if forecast_df is None or forecast_df.empty:
            return pd.DataFrame()
        
        adjusted_forecast = forecast_df.copy()
        
        if 'predicted_price' in adjusted_forecast.columns:
            adjusted_forecast['predicted_price'] = adjusted_forecast['predicted_price'].astype(np.float64)
        
        signal_dates = signals_df[signals_df['signal'] != 0]
        
        if not signal_dates.empty:
            for idx, row in signal_dates.iterrows():
                try:
                    date = row['Date']
                    signal = int(row['signal'])
                    score = float(row['anomaly_score'])
                    
                    future_mask = adjusted_forecast['Date'] >= date
                    future_idx = adjusted_forecast[future_mask].index.tolist()
                    
                    if len(future_idx) > 0:
                        for i, future_i in enumerate(future_idx):
                            decay = np.exp(-0.1 * i)
                            adjustment = signal * score * weight * decay
                            
                            current_pred = float(adjusted_forecast.loc[future_i, 'predicted_price'])
                            new_pred = current_pred * (1 + adjustment/100)
                            adjusted_forecast.loc[future_i, 'predicted_price'] = np.float64(new_pred)
                            
                except (ValueError, TypeError, KeyError) as e:
                    print(f"予測補正エラー: {e}")
                    continue
        
        return adjusted_forecast

# カスタムCSS
CUSTOM_CSS = """
<style>
/* メインコンテナのスタイリング */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

/* ヘッダースタイル */
.header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}

.header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
    font-weight: 700;
}

.header p {
    font-size: 1.2rem;
    opacity: 0.9;
    margin: 0;
}

/* カードスタイル */
.config-card, .results-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border: 1px solid #e1e8ed;
}

.config-card h3, .results-card h3 {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-bottom: 20px;
    font-weight: 600;
}

/* ボタンコンテナ */
.button-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 30px 0;
    margin: 20px 0;
}

/* ボタンスタイル */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    padding: 18px 40px;
    border-radius: 12px;
    font-size: 1.2rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    min-width: 280px;
    text-align: center;
    display: inline-block;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}

.primary-btn:active {
    transform: translateY(0px);
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
}

.secondary-btn {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    border: none;
    color: white;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    width: 100%;
    margin: 10px 0;
}

.secondary-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

/* ステータス表示 */
.status-success {
    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    margin: 15px 0;
}

.status-error {
    background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    margin: 15px 0;
}

.status-warning {
    background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    margin: 15px 0;
}

/* プログレスバー */
.progress-container {
    background: #f0f0f0;
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    text-align: center;
}

.progress-bar {
    width: 100%;
    height: 8px;
    background: #e0e0e0;
    border-radius: 4px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* 結果タブ */
.tab-container {
    background: white;
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* メトリクスカード */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border-left: 4px solid #667eea;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 5px;
}

.metric-label {
    font-size: 0.9rem;
    color: #6c757d;
    font-weight: 500;
}

/* アニメーション */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.6s ease-out;
}

/* レスポンシブデザイン */
@media (max-width: 768px) {
    .header h1 { font-size: 2rem; }
    .header p { font-size: 1rem; }
    .config-card, .results-card { padding: 15px; }
    .metrics-grid { grid-template-columns: 1fr; }
}

/* エージェント結果スタイル */
.agent-result {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    margin-bottom: 20px;
    overflow: hidden;
}

.agent-header {
    background: linear-gradient(135deg, #495057 0%, #343a40 100%);
    color: white;
    padding: 15px 20px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.agent-content {
    padding: 20px;
    border-top: 1px solid #dee2e6;
}

.agent-content pre {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    white-space: pre-wrap;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* 評価結果テーブルスタイル */
.evaluation-table {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin: 15px 0;
}

.evaluation-table table {
    width: 100%;
    border-collapse: collapse;
}

.evaluation-table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px;
    text-align: left;
    font-weight: 600;
}

.evaluation-table td {
    padding: 12px;
    border-bottom: 1px solid #e9ecef;
}

.evaluation-table tr:hover {
    background: #f8f9fa;
}
</style>
"""

# グローバル変数でプログレス状態を管理
progress_state = {"current": 0, "total": 0, "message": ""}

def update_progress(current, total, message):
    """プログレス状態を更新"""
    progress_state["current"] = current
    progress_state["total"] = total
    progress_state["message"] = message

def get_progress_html():
    """プログレスバーのHTMLを生成"""
    if progress_state["total"] == 0:
        return ""
    
    percentage = (progress_state["current"] / progress_state["total"]) * 100
    return f"""
    <div class="progress-container fade-in">
        <h4>分析進行状況</h4>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {percentage}%"></div>
        </div>
        <p>{progress_state['message']} ({progress_state['current']}/{progress_state['total']})</p>
        <p>{percentage:.1f}% 完了</p>
    </div>
    """

# データを準備
def prepare_data(use_sample, file_path=None, symbol='sp500', include_extra_indicators=True, period="2y"):
    """リアルタイム対応データを準備"""
    try:
        if use_sample:
            if data_manager and REALTIME_AVAILABLE:
                # リアルタイムデータを取得
                if include_extra_indicators:
                    # 複数指標のリアルタイムデータを取得
                    data_dict = data_manager.load_multi_indicator_data()
                    
                    # メインデータ（指定されたシンボル、デフォルトはS&P500）
                    main_symbol = symbol if symbol in data_dict else 'sp500'
                    df = data_dict[main_symbol].copy()
                    
                    # 追加指標をマージ
                    if 'volume' in data_dict and not data_dict['volume'].empty:
                        df = pd.merge(df, data_dict['volume'][['Date', 'Volume']], on='Date', how='left')
                    
                    if 'vix' in data_dict and not data_dict['vix'].empty:
                        df = pd.merge(df, data_dict['vix'][['Date', 'VIX']], on='Date', how='left')
                    
                    if 'usdjpy' in data_dict and not data_dict['usdjpy'].empty:
                        df = pd.merge(df, data_dict['usdjpy'][['Date', 'USDJPY']], on='Date', how='left')
                else:
                    # 単一シンボルのデータ
                    df = data_manager.load_sample_data(symbol=symbol)
            else:
                # フォールバック: 既存のサンプルデータ
                from utils.data_utils import load_sample_data, load_multi_indicator_data
                if include_extra_indicators:
                    data_dict = load_multi_indicator_data(START_DATE, END_DATE)
                    df = data_dict['sp500']
                    
                    if 'volume' in data_dict:
                        df = pd.merge(df, data_dict['volume'][['Date', 'Volume']], on='Date', how='left')
                    
                    if 'vix' in data_dict:
                        df = pd.merge(df, data_dict['vix'][['Date', 'VIX']], on='Date', how='left')
                    
                    if 'usdjpy' in data_dict:
                        df = pd.merge(df, data_dict['usdjpy'][['Date', 'USDJPY']], on='Date', how='left')
                else:
                    df = load_sample_data(START_DATE, END_DATE)
        else:
            # ファイルアップロードの場合（既存コードをそのまま維持）
            if file_path is None or file_path == "":
                raise ValueError("ファイルがアップロードされていません")
            
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("対応していないファイル形式です。CSVまたはExcelファイルをアップロードしてください。")
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            
            if 'Close' not in df.columns:
                df.rename(columns={df.columns[1]: 'Close'}, inplace=True)
        
        # データ検証と前処理
        if df.empty:
            raise ValueError("有効なデータが取得できませんでした")
        
        # 日付でソート
        df = df.sort_values('Date').reset_index(drop=True)
        
        # データ型の確認と修正
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        
        if df.empty:
            raise ValueError("有効な価格データがありません")
        
        print(f"✅ データ準備完了: {len(df)}件のデータ")
        if not df.empty:
            print(f"📅 期間: {df['Date'].min().date()} - {df['Date'].max().date()}")
        
        return df
        
    except Exception as e:
        print(f"⚠️ データ準備エラー: {e}")
        # 最終フォールバック
        if data_manager:
            return data_manager._generate_fallback_data()
        else:
            # 緊急時の最小限データ
            dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
            prices = np.random.normal(4000, 200, len(dates))
            return pd.DataFrame({'Date': dates, 'Close': prices})

# 修正版Deep SVDD検出器（エラー対応）
class FixedDeepSVDDDetector:
    """Deep SVDD による異常検知（エラー対応版）"""
    
    def __init__(self, threshold=0.9, epochs=20, batch_size=32, random_state=42):
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        
    def detect(self, data, date_column='Date', value_column='Close', extra_features=None):
        """時系列データ内の異常を検出"""
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
            
            df = data.copy()
            
            # データの前処理と検証
            if df.empty or len(df) < 20:
                print(f"警告: Deep SVDD用データが不足しています（{len(df)}件）")
                return pd.DataFrame()
            
            # 基本的な特徴量を作成
            df['returns'] = df[value_column].pct_change().fillna(0)
            df['log_returns'] = np.log1p(df['returns'])
            
            # 移動平均関連の特徴量（ウィンドウサイズを調整）
            window_size = min(5, len(df) // 4)
            df['rolling_mean'] = df[value_column].rolling(window=window_size, min_periods=1).mean()
            df['rolling_std'] = df[value_column].rolling(window=window_size, min_periods=1).std()
            df['rolling_z'] = (df[value_column] - df['rolling_mean']) / (df['rolling_std'] + 1e-8)
            
            # より小さなウィンドウサイズで移動平均
            ma_window = min(10, len(df) // 2)
            df['ma'] = df[value_column].rolling(window=ma_window, min_periods=1).mean()
            df['ma_ratio'] = df[value_column] / (df['ma'] + 1e-8)
            
            # 特徴量リストを作成
            features = ['returns', 'log_returns', 'rolling_z', 'ma_ratio']
            
            # 追加の特徴量があれば追加
            if extra_features:
                for feature in extra_features:
                    if feature in df.columns:
                        features.append(feature)
            
            # NaN値を処理
            df_features = df.dropna()
            
            if df_features.empty or len(df_features) < 10:
                print(f"警告: Deep SVDD特徴量処理後のデータが不足しています（{len(df_features)}件）")
                return pd.DataFrame()
            
            # 特徴量を抽出
            X = df_features[features].values
            
            # データの検証
            if X.shape[0] == 0 or X.shape[1] == 0:
                print("警告: Deep SVDD特徴量配列が空です")
                return pd.DataFrame()
            
            # 無限値やNaN値をチェック
            if not np.isfinite(X).all():
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 特徴量をスケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 簡単なオートエンコーダーを作成（Deep SVDDの簡易版）
            input_dim = X_scaled.shape[1]
            
            # TensorFlowのシードを設定
            tf.random.set_seed(self.random_state)
            
            # シンプルなオートエンコーダー
            encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(max(2, input_dim // 2), activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dense(2, activation='linear')  # 2次元の潜在空間
            ])
            
            decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(max(2, input_dim // 2), activation='relu', input_shape=(2,)),
                tf.keras.layers.Dense(input_dim, activation='linear')
            ])
            
            # オートエンコーダーを結合
            autoencoder = tf.keras.Sequential([encoder, decoder])
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # 学習（バッチサイズを調整）
            batch_size = min(self.batch_size, len(X_scaled) // 2)
            if batch_size < 1:
                batch_size = 1
                
            history = autoencoder.fit(
                X_scaled, X_scaled,
                epochs=self.epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.1 if len(X_scaled) > 10 else 0
            )
            
            # 再構成エラーを計算
            X_pred = autoencoder.predict(X_scaled, verbose=0)
            reconstruction_errors = np.mean(np.square(X_scaled - X_pred), axis=1)
            
            # 異常判定のための閾値を計算
            error_threshold = np.percentile(reconstruction_errors, self.threshold * 100)
            
            # 結果をデータフレームに追加
            df_features['anomaly_score'] = reconstruction_errors
            df_features['is_anomaly'] = reconstruction_errors > error_threshold
            
            # 変化率を追加
            df_features['pct_change'] = df_features[value_column].pct_change() * 100
            
            # 異常のみを抽出
            anomalies = df_features[df_features['is_anomaly']].copy()
            
            return anomalies
            
        except Exception as e:
            print(f"Deep SVDD検出中にエラーが発生しました: {e}")
            return pd.DataFrame()

# 修正版Isolation Forest検出器（エラー対応強化）
class FixedIsolationForestDetector:
    """Isolation Forest による異常検知（エラー対応強化版）"""
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        
    def detect(self, data, date_column='Date', value_column='Close', extra_features=None):
        """時系列データ内の異常を検出"""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            df = data.copy()
            
            # データの前処理と検証（閾値を緩和）
            if df.empty or len(df) < 10:
                print(f"警告: Isolation Forest用データが不足しています（{len(df)}件）")
                return pd.DataFrame()
            
            # 基本的な特徴量を作成
            df['returns'] = df[value_column].pct_change().fillna(0)
            df['log_returns'] = np.log1p(df['returns'])
            
            # 移動平均関連の特徴量（ウィンドウサイズを動的調整）
            window_5 = min(5, max(2, len(df) // 10))
            window_20 = min(20, max(5, len(df) // 5))
            window_50 = min(50, max(10, len(df) // 2))
            
            df['rolling_mean_5'] = df[value_column].rolling(window=window_5, min_periods=1).mean()
            df['rolling_std_5'] = df[value_column].rolling(window=window_5, min_periods=1).std()
            df['rolling_z'] = (df[value_column] - df['rolling_mean_5']) / (df['rolling_std_5'] + 1e-8)
            
            df['ma_20'] = df[value_column].rolling(window=window_20, min_periods=1).mean()
            df['ma_50'] = df[value_column].rolling(window=window_50, min_periods=1).mean()
            df['ma_ratio_20'] = df[value_column] / (df['ma_20'] + 1e-8)
            df['ma_ratio_50'] = df[value_column] / (df['ma_50'] + 1e-8)
            
            # 特徴量リストを作成
            features = [
                'returns', 'log_returns', 'rolling_z',
                'ma_ratio_20', 'ma_ratio_50'
            ]
            
            # 追加の特徴量があれば追加
            if extra_features:
                for feature in extra_features:
                    if feature in df.columns:
                        features.append(feature)
            
            # NaN値を処理
            df_features = df.dropna()
            
            if df_features.empty or len(df_features) < 5:
                print(f"警告: Isolation Forest特徴量処理後のデータが不足しています（{len(df_features)}件）")
                return pd.DataFrame()
            
            # 特徴量を抽出
            X = df_features[features].values
            
            # データの検証
            if X.shape[0] == 0 or X.shape[1] == 0:
                print("警告: Isolation Forest特徴量配列が空です")
                return pd.DataFrame()
            
            # 無限値やNaN値をチェック
            if not np.isfinite(X).all():
                print("警告: Isolation Forest特徴量に無限値またはNaN値が含まれています")
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 特徴量をスケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # contamination値を調整（データサイズに応じて）
            effective_contamination = min(self.contamination, 0.5)  # 最大50%
            if len(X_scaled) < 20:
                effective_contamination = min(0.1, 1.0 / len(X_scaled))  # 小さなデータセットでは低い値
            
            # Isolation Forestモデルを作成・適用
            self.model = IsolationForest(
                contamination=effective_contamination,
                n_estimators=min(self.n_estimators, 50),  # 計算量を削減
                random_state=self.random_state,
                n_jobs=1  # 並列処理を無効化（安定性向上）
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
            
            print(f"Isolation Forest完了: 入力 {len(df_features)}件, 異常 {len(anomalies)}件")
            return anomalies
            
        except Exception as e:
            print(f"Isolation Forest検出中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    """Isolation Forest による異常検知（エラー対応版）"""
    
    def __init__(self, contamination=0.05, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        
    def detect(self, data, date_column='Date', value_column='Close', extra_features=None):
        """時系列データ内の異常を検出"""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        df = data.copy()
        
        # データの前処理と検証（閾値を緩和）
        if df.empty or len(df) < 5:  # 10 → 5に変更
            print(f"警告: Isolation Forest用データが不足しています（{len(df)}件）")
            return pd.DataFrame()
        
        # 基本的な特徴量を作成
        try:
            df['returns'] = df[value_column].pct_change()
            df['log_returns'] = np.log1p(df['returns'].fillna(0))
            
            # 移動平均関連の特徴量
            df['rolling_mean_5'] = df[value_column].rolling(window=5, min_periods=1).mean()
            df['rolling_std_5'] = df[value_column].rolling(window=5, min_periods=1).std()
            df['rolling_z'] = (df[value_column] - df['rolling_mean_5']) / (df['rolling_std_5'] + 1e-8)
            
            df['ma_20'] = df[value_column].rolling(window=20, min_periods=1).mean()
            df['ma_50'] = df[value_column].rolling(window=50, min_periods=1).mean()
            df['ma_ratio_20'] = df[value_column] / (df['ma_20'] + 1e-8)
            df['ma_ratio_50'] = df[value_column] / (df['ma_50'] + 1e-8)
            
            # 特徴量リストを作成
            features = [
                'returns', 'log_returns', 'rolling_z',
                'ma_ratio_20', 'ma_ratio_50'
            ]
            
            # 追加の特徴量があれば追加
            if extra_features:
                for feature in extra_features:
                    if feature in df.columns:
                        features.append(feature)
            
            # NaN値を処理
            df_features = df.dropna()
            
            if df_features.empty or len(df_features) < 3:  # 5 → 3に変更
                print(f"警告: Isolation Forest特徴量処理後のデータが不足しています（{len(df_features)}件）")
                return pd.DataFrame()
            
            # 特徴量を抽出
            X = df_features[features].values
            
            # データの検証
            if X.shape[0] == 0 or X.shape[1] == 0:
                print("警告: 特徴量配列が空です")
                return pd.DataFrame()
            
            # 無限値やNaN値をチェック
            if not np.isfinite(X).all():
                print("警告: 特徴量に無限値またはNaN値が含まれています")
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # 特徴量をスケーリング
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
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
            
        except Exception as e:
            print(f"Isolation Forest検出中にエラーが発生しました: {e}")
            return pd.DataFrame()

# 異常検知を実行（堅牢版）
def detect_anomalies(df, method, threshold, extra_indicators=None):
    """異常検知を実行（堅牢版）"""
    try:
        method_params = ANOMALY_PARAMS.get(method, {})
        
        # データフレームのコピーを作成し、データ型を統一
        df_copy = df.copy()
        
        # 基本的なデータ検証（閾値を緩和）
        if df_copy.empty or len(df_copy) < 5:  # 10 → 5に変更
            print(f"警告: データが不足しています（{len(df_copy)}件）")
            return pd.DataFrame()
        
        # 数値カラムをfloat64に統一
        numeric_columns = ['Close']
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                df_copy[col] = df_copy[col].astype(np.float64)
        
        # NaN値を除去
        original_length = len(df_copy)
        df_copy = df_copy.dropna(subset=['Close'])
        final_length = len(df_copy)
        
        if df_copy.empty or len(df_copy) < 3:  # 5 → 3に変更
            print(f"警告: 有効なデータが不足しています（{final_length}件、元: {original_length}件）")
            return pd.DataFrame()
        
        print(f"異常検知実行: 手法 {method}, データ数: {len(df_copy)}件")
        
        extra_features = None
        if extra_indicators:
            extra_features = []
            if 'Volume' in df_copy.columns:
                df_copy['Volume'] = pd.to_numeric(df_copy['Volume'], errors='coerce').fillna(0)
                df_copy['Volume'] = df_copy['Volume'].astype(np.float64)
                volume_ma = df_copy['Volume'].rolling(window=20, min_periods=1).mean()
                df_copy['volume_ratio'] = df_copy['Volume'] / (volume_ma + 1e-8)
                df_copy['volume_ratio'] = df_copy['volume_ratio'].astype(np.float64)
                extra_features.append('volume_ratio')
            
            if 'VIX' in df_copy.columns:
                df_copy['VIX'] = pd.to_numeric(df_copy['VIX'], errors='coerce').fillna(20.0)
                df_copy['VIX'] = df_copy['VIX'].astype(np.float64)
                extra_features.append('VIX')
            
            if 'USDJPY' in df_copy.columns:
                df_copy['USDJPY'] = pd.to_numeric(df_copy['USDJPY'], errors='coerce').fillna(110.0)
                df_copy['USDJPY'] = df_copy['USDJPY'].astype(np.float64)
                df_copy['usdjpy_pct_change'] = df_copy['USDJPY'].pct_change() * 100
                df_copy['usdjpy_pct_change'] = df_copy['usdjpy_pct_change'].astype(np.float64)
                extra_features.append('usdjpy_pct_change')
        
        # 機械学習手法の場合は修正版検出器を使用
        if method == 'isolation_forest':
            detector = FixedIsolationForestDetector(
                contamination=method_params.get('contamination', 0.05),
                n_estimators=method_params.get('n_estimators', 100),
                random_state=method_params.get('random_state', 42)
            )
            return detector.detect(df_copy, extra_features=extra_features)
        elif method == 'deep_svdd':
            detector = FixedDeepSVDDDetector(
                threshold=method_params.get('threshold', 0.95),
                epochs=method_params.get('epochs', 20),  # エポック数を削減
                batch_size=method_params.get('batch_size', 32),
                random_state=method_params.get('random_state', 42)
            )
            return detector.detect(df_copy, extra_features=extra_features)
        else:
            # 従来の検出器を使用
            detector = AnomalyDetector(method=method, threshold=float(threshold), **method_params)
            return detector.detect(df_copy, extra_features=extra_features)
        
    except Exception as e:
        print(f"異常検知でエラーが発生しました: {e}")
        return pd.DataFrame()
    """異常検知を実行（堅牢版）"""
    try:
        method_params = ANOMALY_PARAMS.get(method, {})
        
        # データフレームのコピーを作成し、データ型を統一
        df_copy = df.copy()
        
        # 基本的なデータ検証
        if df_copy.empty or len(df_copy) < 10:
            print("警告: データが不足しています")
            return pd.DataFrame()
        
        # 数値カラムをfloat64に統一
        numeric_columns = ['Close']
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                df_copy[col] = df_copy[col].astype(np.float64)
        
        # NaN値を除去
        df_copy = df_copy.dropna(subset=['Close'])
        
        if df_copy.empty or len(df_copy) < 5:
            print("警告: 有効なデータが不足しています")
            return pd.DataFrame()
        
        extra_features = None
        if extra_indicators:
            extra_features = []
            if 'Volume' in df_copy.columns:
                df_copy['Volume'] = pd.to_numeric(df_copy['Volume'], errors='coerce').fillna(0)
                df_copy['Volume'] = df_copy['Volume'].astype(np.float64)
                volume_ma = df_copy['Volume'].rolling(window=20, min_periods=1).mean()
                df_copy['volume_ratio'] = df_copy['Volume'] / (volume_ma + 1e-8)
                df_copy['volume_ratio'] = df_copy['volume_ratio'].astype(np.float64)
                extra_features.append('volume_ratio')
            
            if 'VIX' in df_copy.columns:
                df_copy['VIX'] = pd.to_numeric(df_copy['VIX'], errors='coerce').fillna(20.0)
                df_copy['VIX'] = df_copy['VIX'].astype(np.float64)
                extra_features.append('VIX')
            
            if 'USDJPY' in df_copy.columns:
                df_copy['USDJPY'] = pd.to_numeric(df_copy['USDJPY'], errors='coerce').fillna(110.0)
                df_copy['USDJPY'] = df_copy['USDJPY'].astype(np.float64)
                df_copy['usdjpy_pct_change'] = df_copy['USDJPY'].pct_change() * 100
                df_copy['usdjpy_pct_change'] = df_copy['usdjpy_pct_change'].astype(np.float64)
                extra_features.append('usdjpy_pct_change')
        
        # 機械学習手法の場合は修正版検出器を使用
        if method == 'isolation_forest':
            detector = FixedIsolationForestDetector(
                contamination=method_params.get('contamination', 0.05),
                n_estimators=method_params.get('n_estimators', 100),
                random_state=method_params.get('random_state', 42)
            )
            return detector.detect(df_copy, extra_features=extra_features)
        else:
            # 従来の検出器を使用
            detector = AnomalyDetector(method=method, threshold=float(threshold), **method_params)
            return detector.detect(df_copy, extra_features=extra_features)
        
    except Exception as e:
        print(f"異常検知でエラーが発生しました: {e}")
        return pd.DataFrame()
    """異常検知を実行（データ型問題修正版）"""
    method_params = ANOMALY_PARAMS.get(method, {})
    
    # データフレームのコピーを作成し、データ型を統一
    df_copy = df.copy()
    
    # 数値カラムをfloat64に統一
    numeric_columns = ['Close']
    for col in numeric_columns:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(np.float64)
    
    extra_features = None
    if extra_indicators:
        extra_features = []
        if 'Volume' in df_copy.columns:
            df_copy['Volume'] = df_copy['Volume'].astype(np.float64)
            df_copy['volume_ratio'] = df_copy['Volume'] / df_copy['Volume'].rolling(window=20).mean()
            df_copy['volume_ratio'] = df_copy['volume_ratio'].astype(np.float64)
            extra_features.append('volume_ratio')
        
        if 'VIX' in df_copy.columns:
            df_copy['VIX'] = df_copy['VIX'].astype(np.float64)
            extra_features.append('VIX')
        
        if 'USDJPY' in df_copy.columns:
            df_copy['USDJPY'] = df_copy['USDJPY'].astype(np.float64)
            df_copy['usdjpy_pct_change'] = df_copy['USDJPY'].pct_change() * 100
            df_copy['usdjpy_pct_change'] = df_copy['usdjpy_pct_change'].astype(np.float64)
            extra_features.append('usdjpy_pct_change')
    
    try:
        detector = AnomalyDetector(method=method, threshold=float(threshold), **method_params)
        anomalies = detector.detect(df_copy, extra_features=extra_features)
        
        # 異常検知結果のデータ型も統一
        if not anomalies.empty:
            for col in ['Close', 'pct_change']:
                if col in anomalies.columns:
                    anomalies[col] = anomalies[col].astype(np.float64)
        
        return anomalies
    except Exception as e:
        print(f"異常検知エラー: {e}")
        # エラーの場合は空のDataFrameを返す
        return pd.DataFrame()

# 改善されたプロット作成
def create_enhanced_plot(df, anomalies):
    """改善された時系列データと異常値のプロット（データ型問題対応）"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('価格データと検出された異常', '日次変化率'),
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )
        
        # データ型の安全な変換
        df_safe = df.copy()
        if 'Close' in df_safe.columns:
            df_safe['Close'] = pd.to_numeric(df_safe['Close'], errors='coerce')
        
        # メインの価格データ
        fig.add_trace(
            go.Scatter(
                x=df_safe['Date'],
                y=df_safe['Close'],
                mode='lines',
                name='価格',
                line=dict(color='#2E86AB', width=2),
                hovertemplate='<b>日付:</b> %{x}<br><b>価格:</b> ¥%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 異常値
        if not anomalies.empty:
            anomalies_safe = anomalies.copy()
            if 'Close' in anomalies_safe.columns:
                anomalies_safe['Close'] = pd.to_numeric(anomalies_safe['Close'], errors='coerce')
            if 'pct_change' in anomalies_safe.columns:
                anomalies_safe['pct_change'] = pd.to_numeric(anomalies_safe['pct_change'], errors='coerce')
            else:
                anomalies_safe['pct_change'] = 0.0
                
            fig.add_trace(
                go.Scatter(
                    x=anomalies_safe['Date'],
                    y=anomalies_safe['Close'],
                    mode='markers',
                    name='検出された異常',
                    marker=dict(
                        color='#E63946',
                        size=12,
                        symbol='diamond',
                        line=dict(color='white', width=2)
                    ),
                    hovertemplate='<b>異常日:</b> %{x}<br><b>価格:</b> ¥%{y:,.0f}<br><b>変化率:</b> %{customdata:.2f}%<extra></extra>',
                    customdata=anomalies_safe['pct_change'].fillna(0)
                ),
                row=1, col=1
            )
        
        # 変化率の計算と表示
        if 'pct_change' not in df_safe.columns:
            df_safe['pct_change'] = df_safe['Close'].pct_change() * 100
        
        df_safe['pct_change'] = pd.to_numeric(df_safe['pct_change'], errors='coerce').fillna(0)
        
        # 変化率プロット
        colors = ['#E63946' if abs(float(x)) > 2 else '#2E86AB' for x in df_safe['pct_change']]
        
        fig.add_trace(
            go.Bar(
                x=df_safe['Date'],
                y=df_safe['pct_change'],
                name='日次変化率',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='<b>日付:</b> %{x}<br><b>変化率:</b> %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(
                text='<b>時系列データ分析結果</b>',
                x=0.5,
                font=dict(size=20, color='#2c3e50')
            ),
            showlegend=True,
            height=700,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="日付", row=2, col=1)
        fig.update_yaxes(title_text="価格", row=1, col=1)
        fig.update_yaxes(title_text="変化率 (%)", row=2, col=1)
        
        return fig
        
    except Exception as e:
        print(f"プロット作成エラー: {e}")
        # エラーの場合は空のプロットを返す
        return go.Figure().add_annotation(text="プロット作成中にエラーが発生しました", 
                                        x=0.5, y=0.5, showarrow=False)

# 予測プロット作成
def create_enhanced_forecast_plot(df, anomalies, forecast_df, adjusted_forecast_df):
    """改善された予測プロット"""
    fig = go.Figure()
    
    # 実際の価格（最近のデータのみ表示）
    recent_data = df.tail(100)
    fig.add_trace(go.Scatter(
        x=recent_data['Date'],
        y=recent_data['Close'],
        mode='lines',
        name='実際の価格',
        line=dict(color='#2E86AB', width=3),
        hovertemplate='<b>日付:</b> %{x}<br><b>実際価格:</b> ¥%{y:,.0f}<extra></extra>'
    ))
    
    # 異常値
    if not anomalies.empty:
        recent_anomalies = anomalies[anomalies['Date'] >= recent_data['Date'].min()]
        if not recent_anomalies.empty:
            fig.add_trace(go.Scatter(
                x=recent_anomalies['Date'],
                y=recent_anomalies['Close'],
                mode='markers',
                name='検出された異常',
                marker=dict(
                    color='#E63946',
                    size=12,
                    symbol='diamond',
                    line=dict(color='white', width=2)
                )
            ))
    
    # 基本予測
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['predicted_price'],
            mode='lines',
            name='基本予測',
            line=dict(color='#F77F00', width=2, dash='dash'),
            hovertemplate='<b>予測日:</b> %{x}<br><b>基本予測:</b> ¥%{y:,.0f}<extra></extra>'
        ))
    
    # 異常調整済み予測
    if adjusted_forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=adjusted_forecast_df['Date'],
            y=adjusted_forecast_df['predicted_price'],
            mode='lines',
            name='異常調整済み予測',
            line=dict(color='#6A994E', width=3),
            hovertemplate='<b>予測日:</b> %{x}<br><b>調整済み予測:</b> ¥%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='<b>価格予測分析</b>',
            x=0.5,
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title="日付",
        yaxis_title="価格",
        template='plotly_white',
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# LLMクライアントを取得
def get_llm_client(provider):
    """指定されたプロバイダに基づいてLLMクライアントを取得"""
    if provider == "openai":
        return OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    elif provider == "huggingface":
        return HuggingFaceClient(api_key=HF_API_KEY, model=HF_MODEL)
    else:
        return MockLLMClient()

# エージェントシステムを実行
def run_agent_system(anomalies, llm_provider, enabled_agents, reference_data=None):
    """エージェントシステムを実行して結果を返す"""
    llm_client = get_llm_client(llm_provider)
    
    # 重要な異常のみに絞る
    if len(anomalies) > 3:
        anomalies = anomalies.iloc[anomalies['pct_change'].abs().argsort()[::-1][:3]].copy()
    
    # エージェントを初期化
    agents = []
    agent_names = []
    
    if 'web' in enabled_agents:
        agents.append(WebInformationAgent(llm_client=llm_client))
        agent_names.append('Web情報エージェント')
    
    if 'knowledge' in enabled_agents:
        agents.append(KnowledgeBaseAgent(llm_client=llm_client))
        agent_names.append('知識ベースエージェント')
    
    if 'crosscheck' in enabled_agents:
        agents.append(CrossCheckAgent(llm_client=llm_client))
        agent_names.append('クロスチェックエージェント')
    
    # コンテキスト準備
    context = {
        'reference_data': reference_data if reference_data else {}
    }
    
    # エージェント結果を保存
    agent_findings = {}
    
    # 各エージェントを実行（プログレス更新付き）
    total_agents = len(agents) + (2 if 'report' in enabled_agents or 'manager' in enabled_agents else 0)
    current_agent = 0
    
    for i, agent in enumerate(agents):
        if agent.name in ["レポート統合エージェント", "管理者エージェント"]:
            continue
        
        current_agent += 1
        update_progress(current_agent, total_agents, f"{agent.name}を実行中...")
        time.sleep(0.5)  # プログレス表示のため
        
        agent_result = agent.process(anomalies, context)
        agent_findings[agent.name] = agent_result
    
    # 統合エージェントを実行
    if 'report' in enabled_agents:
        current_agent += 1
        update_progress(current_agent, total_agents, "レポート統合エージェントを実行中...")
        time.sleep(0.5)
        
        report_agent = ReportIntegrationAgent(llm_client=llm_client)
        context['agent_findings'] = agent_findings
        report_agent_result = report_agent.process(anomalies, context)
        agent_findings[report_agent.name] = report_agent_result
    
    # 管理者エージェントを実行
    if 'manager' in enabled_agents:
        current_agent += 1
        update_progress(current_agent, total_agents, "管理者エージェントを実行中...")
        time.sleep(0.5)
        
        manager_agent = ManagerAgent(llm_client=llm_client)
        manager_agent_result = manager_agent.process(anomalies, context)
        agent_findings[manager_agent.name] = manager_agent_result
    
    update_progress(total_agents, total_agents, "分析完了!")
    return agent_findings

# メトリクスカードのHTML生成
def create_metrics_cards(anomalies_count, total_data_points, detection_rate):
    """メトリクスカードのHTMLを生成"""
    return f"""
    <div class="metrics-grid fade-in">
        <div class="metric-card">
            <div class="metric-value">{anomalies_count}</div>
            <div class="metric-label">検出された異常数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_data_points:,}</div>
            <div class="metric-label">総データポイント数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{detection_rate:.3f}%</div>
            <div class="metric-label">異常検出率</div>
        </div>
    </div>
    """

# 分析を実行
def run_analysis(data_source, file_path, detection_method, threshold, llm_provider, 
                use_web_agent, use_knowledge_agent, use_crosscheck_agent, 
                use_report_agent, use_manager_agent, include_extra_indicators=True,
                generate_signals=True, forecast_days=30, progress=gr.Progress()):
    try:
        # プログレス初期化
        update_progress(0, 8, "分析を開始しています...")
        progress(0.0, desc="分析を開始しています...")
        
        # データを準備
        update_progress(1, 8, "データを準備中...")
        progress(0.125, desc="データを準備中...")
        use_sample = (data_source == "sample")
        df = prepare_data(use_sample, file_path, include_extra_indicators)
        
        # 異常検知を実行
        update_progress(2, 8, "異常検知を実行中...")
        progress(0.25, desc="異常検知を実行中...")
        anomalies = detect_anomalies(df, detection_method, threshold, include_extra_indicators)
        
        # プロットを作成
        update_progress(3, 8, "可視化を作成中...")
        progress(0.375, desc="可視化を作成中...")
        plot = create_enhanced_plot(df, anomalies)
        
        # 基本的な統計情報
        anomalies_count = len(anomalies)
        total_data_points = len(df)
        detection_rate = (anomalies_count / total_data_points) * 100 if total_data_points > 0 else 0
        
        # メトリクスカードを作成
        metrics_html = create_metrics_cards(anomalies_count, total_data_points, detection_rate)
        
        # 異常データを表として整形
        if anomalies.empty:
            anomalies_df = pd.DataFrame(columns=['日付', '値', '変化率 (%)'])
        else:
            anomalies_df = anomalies.copy()
            anomalies_df = anomalies_df[['Date', 'Close', 'pct_change']]
            anomalies_df.columns = ['日付', '値', '変化率 (%)']
            anomalies_df['日付'] = anomalies_df['日付'].dt.strftime('%Y-%m-%d')
            anomalies_df['変化率 (%)'] = anomalies_df['変化率 (%)'].round(2)
        
        # 有効なエージェントのリストを作成
        enabled_agents = []
        if use_web_agent:
            enabled_agents.append('web')
        if use_knowledge_agent:
            enabled_agents.append('knowledge')
        if use_crosscheck_agent:
            enabled_agents.append('crosscheck')
        if use_report_agent:
            enabled_agents.append('report')
        if use_manager_agent:
            enabled_agents.append('manager')
        
        # 参照データを準備
        reference_data = {}
        if include_extra_indicators:
            if 'Volume' in df.columns:
                volume_df = df[['Date', 'Volume']].copy()
                reference_data['volume'] = volume_df
            
            if 'VIX' in df.columns:
                vix_df = df[['Date', 'VIX']].copy()
                reference_data['vix'] = vix_df
            
            if 'USDJPY' in df.columns:
                usdjpy_df = df[['Date', 'USDJPY']].copy()
                reference_data['usdjpy'] = usdjpy_df
        
        # エージェント分析を実行
        update_progress(4, 8, "エージェント分析を実行中...")
        progress(0.5, desc="エージェント分析を実行中...")
        
        agent_findings = {}
        if enabled_agents and not anomalies.empty:
            agent_findings = run_agent_system(anomalies, llm_provider, enabled_agents, reference_data)
        
        # シグナル生成と予測
        update_progress(6, 8, "予測モデルを実行中...")
        progress(0.75, desc="予測モデルを実行中...")
        
        signals_df = None
        forecast_df = None
        adjusted_forecast_df = None
        forecast_plot = None
        
        if generate_signals and not anomalies.empty:
            # 修正版SignalGeneratorを使用
            signal_generator = FixedSignalGenerator(
                threshold=SIGNAL_THRESHOLD, 
                window_size=SIGNAL_WINDOW
            )
            
            signals_df = signal_generator.generate_signals(df, anomalies)
            
            # 予測パイプライン
            pipeline = ForecastingPipeline(
                forecasting_model=FORECASTING_MODEL,
                signal_threshold=SIGNAL_THRESHOLD,
                window_size=SIGNAL_WINDOW,
                lookback=FORECASTING_PARAMS[FORECASTING_MODEL].get('lookback', 60)
            )
            
            # 予測のみ実行（シグナル生成は既に完了）
            _, forecast_df, _ = pipeline.process(
                df, anomalies, 
                train_model=True, 
                days_ahead=forecast_days, 
                adjustment_weight=0.5
            )
            
            # 予測補正を手動実行
            adjusted_forecast_df = signal_generator.forecast_adjustment(
                signals_df, forecast_df, weight=0.5
            )
            
            forecast_plot = create_enhanced_forecast_plot(df, anomalies, forecast_df, adjusted_forecast_df)
        
        # シグナルを表として整形
        signals_table = pd.DataFrame()
        if signals_df is not None and not signals_df.empty:
            signals_table = signals_df[signals_df['signal'] != 0][['Date', 'Close', 'pct_change', 'signal', 'anomaly_score']]
            if not signals_table.empty:
                signals_table.columns = ['日付', '価格', '変化率 (%)', 'シグナル(-1=売/1=買)', '異常強度']
                signals_table['日付'] = signals_table['日付'].dt.strftime('%Y-%m-%d')
                signals_table['変化率 (%)'] = signals_table['変化率 (%)'].round(2)
                signals_table['異常強度'] = signals_table['異常強度'].round(2)
        
        # 最終更新
        update_progress(8, 8, "分析完了!")
        progress(1.0, desc="分析完了!")
        
        # 成功メッセージ
        status_html = f'<div class="status-success fade-in">✅ 分析が正常に完了しました！{anomalies_count}件の異常を検出しました。</div>'
        
        return (
            plot, forecast_plot, status_html, metrics_html, 
            anomalies_df, signals_table, agent_findings, df, anomalies
        )
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(error_traceback)
        
        error_html = f'<div class="status-error fade-in">❌ エラーが発生しました: {str(e)}</div>'
        empty_metrics = '<div class="metrics-grid"></div>'
        
        return (
            None, None, error_html, empty_metrics,
            pd.DataFrame(), pd.DataFrame(), {}, None, None
        )

# エージェント結果を改善されたHTMLとして整形
def format_agent_findings_enhanced(agent_findings):
    """エージェント結果を改善されたHTMLとして整形"""
    # agent_findingsが空辞書、None、または空のDataFrameかどうかを安全にチェック
    if (agent_findings is None or 
        (isinstance(agent_findings, dict) and len(agent_findings) == 0) or
        (hasattr(agent_findings, 'empty') and agent_findings.empty)):
        return '<div class="status-warning fade-in">⚠️ エージェント分析結果はありません。</div>'
    
    html = '<div class="fade-in">'
    
    # 最終評価が存在する場合はそれを優先表示
    if "管理者エージェント" in agent_findings:
        manager_findings = agent_findings["管理者エージェント"].get("findings", {})
        if manager_findings:
            html += '''
            <div class="agent-result">
                <div class="agent-header">
                    <h3>🎯 最終評価・推奨事項</h3>
                    <span>⭐ エグゼクティブサマリー</span>
                </div>
                <div class="agent-content">
            '''
            
            for date, data in manager_findings.items():
                anomaly_details = data.get('anomaly_details', {})
                pct_change = anomaly_details.get('pct_change', 'N/A')
                
                html += f'''
                <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; border-left: 4px solid #28a745;">
                    <h4 style="color: #155724; margin-bottom: 15px;">📅 {date} (変化率: {pct_change}%)</h4>
                    <pre style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; margin: 0;">{data.get('final_assessment', 'N/A')}</pre>
                </div>
                '''
            
            html += '</div></div>'
    
    # 統合レポートが存在する場合はそれを表示
    if "レポート統合エージェント" in agent_findings:
        report_findings = agent_findings["レポート統合エージェント"].get("findings", {})
        if report_findings:
            html += '''
            <div class="agent-result">
                <div class="agent-header">
                    <h3>📋 統合分析レポート</h3>
                    <span>📊 複数エージェント統合結果</span>
                </div>
                <div class="agent-content">
            '''
            
            for date, data in report_findings.items():
                anomaly_details = data.get('anomaly_details', {})
                pct_change = anomaly_details.get('pct_change', 'N/A')
                
                html += f'''
                <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border-radius: 12px; border-left: 4px solid #856404;">
                    <h4 style="color: #856404; margin-bottom: 15px;">📅 {date} (変化率: {pct_change}%)</h4>
                    <pre style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; margin: 0;">{data.get('integrated_report', 'N/A')}</pre>
                </div>
                '''
            
            html += '</div></div>'
    
    # 個別のエージェント結果も表示
    html += '<h3 style="margin: 30px 0 20px 0; color: #495057;">🔍 個別エージェント分析詳細</h3>'
    
    agent_colors = {
        'Web情報エージェント': '#007bff',
        '知識ベースエージェント': '#28a745', 
        'クロスチェックエージェント': '#ffc107'
    }
    
    agent_icons = {
        'Web情報エージェント': '🌐',
        '知識ベースエージェント': '📚',
        'クロスチェックエージェント': '🔄'
    }
    
    for agent_name, findings in agent_findings.items():
        if agent_name not in ["レポート統合エージェント", "管理者エージェント"]:
            color = agent_colors.get(agent_name, '#6c757d')
            icon = agent_icons.get(agent_name, '🤖')
            
            html += f'''
            <details class="agent-result" style="margin-bottom: 20px;">
                <summary class="agent-header" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                    <span>{icon} {agent_name}</span>
                    <span style="font-size: 0.9em; opacity: 0.9;">クリックして詳細を表示</span>
                </summary>
                <div class="agent-content">
            '''
            
            if "error" in findings:
                html += f'<div class="status-error">❌ エラー: {findings["error"]}</div>'
            elif "findings" in findings:
                for date, data in findings["findings"].items():
                    if "llm_analysis" in data:
                        html += f'''
                        <div style="margin-bottom: 20px;">
                            <h4 style="color: {color}; margin-bottom: 10px;">📅 {date}</h4>
                            <pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 3px solid {color}; margin-bottom: 15px;">{data['llm_analysis']}</pre>
                        '''
                        
                        # クロスチェックエージェントの場合、追加指標も表示
                        if agent_name == "クロスチェックエージェント" and "additional_metrics" in data:
                            html += f'''
                            <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin-top: 10px;">
                                <h5 style="color: #0056b3; margin-bottom: 10px;">📈 追加市場指標</h5>
                                <pre style="background: white; padding: 12px; border-radius: 6px; margin: 0; font-size: 0.9em;">{data['additional_metrics']}</pre>
                            </div>
                            '''
                        
                        html += '</div><hr style="border: none; border-top: 1px solid #dee2e6; margin: 20px 0;">'
            
            html += '</div></details>'
    
    html += '</div>'
    return html

# 評価関数
def run_evaluation(df, anomalies, known_anomalies_str, delay_tolerance):
    try:
        known_anomalies = [date.strip() for date in known_anomalies_str.split(',') if date.strip()]
        
        evaluator = AnomalyEvaluator(known_anomalies)
        eval_results = evaluator.evaluate(df, anomalies, int(delay_tolerance))
        
        # 評価指標をDataFrameに変換
        metrics_df = pd.DataFrame({
            '評価指標': ['適合率 (Precision)', '再現率 (Recall)', 'F1スコア', 'Fβスコア', '検知遅延 (日)'],
            '値': [
                f"{eval_results['precision']:.4f}",
                f"{eval_results['recall']:.4f}",
                f"{eval_results['f1_score']:.4f}",
                f"{eval_results['f_beta_score']:.4f}",
                f"{eval_results['detection_delay'] if eval_results['detection_delay'] is not None else 'N/A'}"
            ],
            '説明': [
                '検出された異常のうち、実際に異常だった割合',
                '実際の異常のうち、検出できた割合',
                '適合率と再現率の調和平均（総合性能指標）',
                '再現率を重視した総合指標（重要な異常の見逃し防止）',
                '実際の異常発生から検出までの平均遅れ時間'
            ]
        })
        
        # 混同行列を作成
        cm_df = pd.DataFrame({
            '実際→': ['正常', '異常'],
            '予測: 正常': [eval_results['true_negatives'], eval_results['false_negatives']],
            '予測: 異常': [eval_results['false_positives'], eval_results['true_positives']]
        })
        
        return metrics_df, cm_df
    
    except Exception as e:
        import traceback
        error_df = pd.DataFrame({'エラー': [str(e), traceback.format_exc()]})
        return error_df, pd.DataFrame()

# 手法比較関数（エラーハンドリング強化版）
def compare_methods(df, methods, thresholds_text, known_anomalies_str):
    try:
        thresholds = [float(t.strip()) for t in thresholds_text.split(',') if t.strip()]
        known_anomalies = [date.strip() for date in known_anomalies_str.split(',') if date.strip()]
        
        evaluator = AnomalyEvaluator(known_anomalies)
        
        # デバッグ: データ情報を表示
        print(f"手法比較開始: データフレーム形状 {df.shape}, カラム: {list(df.columns)}")
        if 'Date' in df.columns and 'Close' in df.columns:
            print(f"日付範囲: {df['Date'].min()} ～ {df['Date'].max()}")
            print(f"価格範囲: {df['Close'].min():.2f} ～ {df['Close'].max():.2f}")
        
        # カスタム比較関数（エラー対応版）
        results = []
        
        for method in methods:
            for threshold in thresholds:
                try:
                    # データの前処理とチェック
                    df_processed = df.copy()
                    
                    # 基本的なデータ検証（閾値を緩和）
                    if df_processed.empty or len(df_processed) < 10:  # 50 → 10に変更
                        print(f"警告: データが不足しています（{len(df_processed)}件）。手法 {method}, 閾値 {threshold} をスキップします。")
                        continue
                    
                    # 数値カラムの確認
                    if 'Close' not in df_processed.columns:
                        print(f"警告: 'Close'カラムが見つかりません。手法 {method}, 閾値 {threshold} をスキップします。")
                        continue
                    
                    # NaN値の処理
                    df_processed['Close'] = pd.to_numeric(df_processed['Close'], errors='coerce')
                    original_length = len(df_processed)
                    df_processed = df_processed.dropna(subset=['Close'])
                    final_length = len(df_processed)
                    
                    if final_length < 5:  # 10 → 5に変更
                        print(f"警告: 有効なデータが不足しています（{final_length}件、元: {original_length}件）。手法 {method}, 閾値 {threshold} をスキップします。")
                        continue
                    
                    # 機械学習手法の場合の追加チェック（条件を緩和）
                    if method in ['isolation_forest', 'deep_svdd']:
                        # 十分なデータポイントがあるかチェック（100 → 30に変更）
                        if len(df_processed) < 30:
                            print(f"警告: 機械学習手法 {method} には最低30データポイントが必要です（現在: {len(df_processed)}件）。スキップします。")
                            continue
                        
                        # 変動性のチェック
                        if df_processed['Close'].std() == 0:
                            print(f"警告: データに変動がありません。手法 {method}, 閾値 {threshold} をスキップします。")
                            continue
                    
                    print(f"実行中: 手法 {method}, 閾値 {threshold}, データ数: {len(df_processed)}件")
                    
                    # 異常検知器を作成（修正版を使用）
                    if method == 'isolation_forest':
                        # 修正版Isolation Forest検出器を直接使用
                        detector_obj = FixedIsolationForestDetector(
                            contamination=0.05,
                            n_estimators=50,  # 計算量削減
                            random_state=42
                        )
                        detected_anomalies = detector_obj.detect(df_processed)
                    elif method == 'deep_svdd':
                        # 修正版Deep SVDD検出器を直接使用
                        detector_obj = FixedDeepSVDDDetector(
                            threshold=0.95,
                            epochs=10,  # エポック数削減（高速化）
                            batch_size=16,
                            random_state=42
                        )
                        detected_anomalies = detector_obj.detect(df_processed)
                    else:
                        # 通常の検出器を使用
                        detector = AnomalyDetector(method=method, threshold=threshold)
                        detected_anomalies = detector.detect(df_processed)
                    
                    # 評価を実行
                    eval_result = evaluator.evaluate(df_processed, detected_anomalies)
                    
                    # 結果を格納
                    results.append({
                        '検出手法': method,
                        '閾値': threshold,
                        '適合率': round(eval_result['precision'], 4),
                        '再現率': round(eval_result['recall'], 4),
                        'F1スコア': round(eval_result['f1_score'], 4),
                        'Fβスコア': round(eval_result['f_beta_score'], 4),
                        '検知数': len(detected_anomalies),
                        '真陽性': eval_result['true_positives'],
                        '偽陽性': eval_result['false_positives'],
                        '偽陰性': eval_result['false_negatives']
                    })
                    
                except Exception as method_error:
                    print(f"手法 {method} (閾値: {threshold}) でエラーが発生しました: {str(method_error)}")
                    # エラーが発生した手法はスキップして続行
                    results.append({
                        '検出手法': method,
                        '閾値': threshold,
                        '適合率': 'エラー',
                        '再現率': 'エラー',
                        'F1スコア': 'エラー',
                        'Fβスコア': 'エラー',
                        '検知数': 'エラー',
                        '真陽性': 'エラー',
                        '偽陽性': 'エラー',
                        '偽陰性': 'エラー'
                    })
                    continue
        
        # 結果をDataFrameに変換
        if results:
            results_df = pd.DataFrame(results)
            return results_df
        else:
            return pd.DataFrame({'メッセージ': ['すべての手法でエラーが発生しました。データを確認してください。']})
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"手法比較でエラーが発生しました: {e}")
        print(error_traceback)
        error_df = pd.DataFrame({
            'エラー': ['手法比較中にエラーが発生しました'],
            '詳細': [str(e)],
            '推奨対処': ['データ形式を確認するか、異なる手法を試してください']
        })
        return error_df

# Gradio UIの作成
def create_gradio_ui():
    with gr.Blocks(
        title="🔍 マルチエージェントLLM異常検知分析システム",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
            neutral_hue="gray"
        )
    ) as app:
        
        # ヘッダー
        gr.HTML(f"""
        <div class="header fade-in">
            <h1>🔍 マルチエージェントLLM異常検知分析システム</h1>
            <p>🤖 AI駆動の時系列データ異常検知と包括的分析プラットフォーム</p>
        </div>
        """)
        
        # データおよび結果を保持する状態
        stored_df = gr.State(None)
        stored_anomalies = gr.State(None)
        
        # 設定セクション
        with gr.Group():
            gr.HTML('<div class="config-card fade-in">')
            gr.HTML('<h3>⚙️ 分析設定</h3>')
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<h4>📊 データ設定</h4>')
                    data_source = gr.Radio(
                        label="📁 データソース",
                        choices=[("サンプルデータを使用", "sample"), ("ファイルをアップロード", "upload")],
                        value="sample",
                        info="🎯 サンプルデータには歴史的な市場異常が含まれています"
                    )
                    file_path = gr.File(
                        label="📋 時系列データファイル (CSV/Excel)",
                        type="filepath",
                        visible=False
                    )
                    include_extra_indicators = gr.Checkbox(
                        label="📈 追加指標を含める（出来高、VIX、ドル円）",
                        value=True,
                        info="✨ より詳細な市場分析のための追加データ"
                    )
                    
                with gr.Column(scale=1):
                    gr.HTML('<h4>🎯 異常検知設定</h4>')
                    detection_method = gr.Dropdown(
                        label="🔍 検出方法",
                        choices=[
                            ("Z-Score（標準偏差ベース）", "z_score"),
                            ("IQR（四分位数ベース）", "iqr"),
                            ("移動平均（トレンド乖離）", "moving_avg"),
                            ("Isolation Forest（機械学習）", "isolation_forest"),
                            ("Deep SVDD（深層学習）", "deep_svdd")
                        ],
                        value="z_score",
                        info="📊 各手法の特徴を理解して選択してください"
                    )
                    threshold = gr.Slider(
                        label="🎚️ 検出閾値",
                        minimum=1.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        info="⚡ 値が大きいほど検出される異常が少なくなります"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<h4>🤖 LLM設定</h4>')
                    llm_provider = gr.Radio(
                        label="🧠 LLMプロバイダ",
                        choices=[
                            ("OpenAI GPT", "openai"),
                            ("Hugging Face", "huggingface"),
                            ("Mock（デモ用）", "mock")
                        ],
                        value="mock",
                        info="🔑 OpenAI/HuggingFaceを使用する場合はAPIキーが必要です"
                    )
                
                with gr.Column(scale=1):
                    gr.HTML('<h4>👥 エージェント設定</h4>')
                    with gr.Row():
                        with gr.Column():
                            use_web_agent = gr.Checkbox(label="🌐 Web情報エージェント", value=True)
                            use_knowledge_agent = gr.Checkbox(label="📚 知識ベースエージェント", value=True)
                            use_crosscheck_agent = gr.Checkbox(label="🔄 クロスチェックエージェント", value=True)
                        with gr.Column():
                            use_report_agent = gr.Checkbox(label="📋 レポート統合エージェント", value=True)
                            use_manager_agent = gr.Checkbox(label="🎯 管理者エージェント", value=True)
            
            with gr.Row():
                with gr.Column():
                    generate_signals = gr.Checkbox(
                        label="📈 売買シグナルと将来予測を生成",
                        value=True,
                        info="🎯 異常から投資シグナルを生成し、価格を予測します"
                    )
                    forecast_days = gr.Slider(
                        label="📅 予測日数",
                        minimum=10,
                        maximum=90,
                        value=30,
                        step=5,
                        info="🔮 将来の何日分を予測するか"
                    )
            
            # データソース変更時のイベントハンドラ
            def update_file_visibility(choice):
                return gr.update(visible=(choice == "upload"))
            
            data_source.change(fn=update_file_visibility, inputs=data_source, outputs=file_path)
            
            gr.HTML('</div>')
        
        # 分析実行ボタン（完璧な中央配置）
        gr.HTML('<div class="button-container">')
        analyze_btn = gr.Button(
            "🚀 異常検知分析を開始",
            variant="primary",
            size="lg",
            elem_classes=["primary-btn"]
        )
        gr.HTML('</div>')
        
        # ステータス表示
        status_display = gr.HTML("")
        metrics_display = gr.HTML("")
        
        # 結果表示セクション
        with gr.Group() as results_container:
            gr.HTML('<div class="results-card fade-in">')
            gr.HTML('<h3>📊 分析結果</h3>')
            
            with gr.Tabs() as result_tabs:
                # メインプロットタブ
                with gr.TabItem("📈 時系列分析", id="main_plot"):
                    plot_output = gr.Plot(label="時系列データと異常検知結果")
                
                # 予測プロットタブ
                with gr.TabItem("🔮 将来予測", id="forecast"):
                    forecast_plot_output = gr.Plot(label="価格予測と異常調整")
                
                # 検出結果タブ
                with gr.TabItem("🎯 検出された異常", id="anomalies"):
                    gr.HTML('<h4>📋 異常検知結果一覧</h4>')
                    anomaly_table = gr.DataFrame(
                        label="検出された異常データ",
                        interactive=False,
                        wrap=True
                    )
                
                # シグナルタブ
                with gr.TabItem("💰 投資シグナル", id="signals"):
                    gr.HTML('<h4>📊 売買推奨シグナル</h4>')
                    signals_table = gr.DataFrame(
                        label="異常から生成された売買シグナル",
                        interactive=False,
                        wrap=True
                    )
                
                # エージェント分析タブ
                with gr.TabItem("🤖 AI分析結果", id="agents"):
                    gr.HTML('<h4>🧠 マルチエージェント総合分析</h4>')
                    agent_results = gr.HTML("")
                
                # 評価タブ
                with gr.TabItem("📊 性能評価", id="evaluation"):
                    gr.HTML('<div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">')
                    gr.HTML('<h4>🎯 異常検知精度の評価</h4>')
                    gr.HTML('<p>📈 既知の異常日付を入力して、検出アルゴリズムの性能を定量的に評価します。</p>')
                    gr.HTML('</div>')
                    
                    with gr.Row():
                        with gr.Column():
                            known_anomalies = gr.Textbox(
                                label="📅 既知の異常日付（カンマ区切り、YYYY-MM-DD形式）",
                                placeholder="例: 1987-10-19, 2008-10-13, 2020-03-16",
                                value="1987-10-19, 2008-10-13, 2020-03-16",
                                info="🎯 評価用の正解データとして使用されます"
                            )
                        with gr.Column():
                            delay_tolerance = gr.Number(
                                label="⏱️ 検知遅延許容値（日数）",
                                value=0,
                                minimum=0,
                                step=1,
                                info="🔍 検知が何日遅れても正解とみなすか"
                            )
                    
                    with gr.Row():
                        evaluate_btn = gr.Button("📊 評価実行", variant="secondary", elem_classes=["secondary-btn"])
                        compare_btn = gr.Button("⚖️ 手法比較", variant="secondary", elem_classes=["secondary-btn"])
                    
                    with gr.Accordion("📚 評価指標の詳細説明", open=False):
                        gr.HTML("""
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;">
                            <h5>📊 性能指標について</h5>
                            <ul style="line-height: 1.8;">
                                <li><strong>🎯 適合率（Precision）:</strong> 検出された異常のうち、実際に異常だった割合。値が高いほど誤検知が少ない。</li>
                                <li><strong>🔍 再現率（Recall）:</strong> 実際の異常のうち、検出できた割合。値が高いほど見逃しが少ない。</li>
                                <li><strong>⚖️ F1スコア:</strong> 適合率と再現率の調和平均。総合的な性能指標。</li>
                                <li><strong>📈 Fβスコア:</strong> 再現率をより重視した指標（β=2）。重大な異常を見逃したくない場合に有用。</li>
                                <li><strong>⏱️ 検知遅延:</strong> 実際の異常発生から検出までの平均遅れ時間（日数）。</li>
                            </ul>
                        </div>
                        """)
                    
                    gr.HTML('<h5>📈 基本評価指標</h5>')
                    evaluation_metrics = gr.DataFrame(
                        label="性能評価結果",
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.HTML('<h5>📊 混同行列（予測精度の詳細）</h5>')
                    confusion_matrix_df = gr.DataFrame(
                        label="混同行列",
                        interactive=False
                    )
                    
                    gr.HTML('<h5>⚖️ 複数手法・閾値の比較分析</h5>')
                    with gr.Row():
                        methods_select = gr.CheckboxGroup(
                            label="🔍 比較する検出手法",
                            choices=[
                                ("Z-Score", "z_score"),
                                ("IQR", "iqr"), 
                                ("移動平均", "moving_avg"),
                                ("Isolation Forest", "isolation_forest"),
                                ("Deep SVDD", "deep_svdd")
                            ],
                            value=["z_score", "iqr", "moving_avg"],
                            info="📋 複数の手法を選択して性能を比較できます"
                        )
                        thresholds_text = gr.Textbox(
                            label="🎚️ 比較する閾値（カンマ区切り）",
                            value="2.0, 2.5, 3.0, 3.5",
                            placeholder="例: 2.0, 2.5, 3.0, 3.5",
                            info="📊 異なる閾値での性能を比較します"
                        )
                    
                    comparison_results = gr.DataFrame(
                        label="手法比較結果",
                        interactive=False,
                        wrap=True
                    )
            
            gr.HTML('</div>')
        
        # イベントハンドラー
        def handle_analyze_click(*args):
            return run_analysis(*args)
        
        def handle_results_display(plot, forecast_plot, status, metrics, anomalies_table, signals_table, findings, df, anomalies):
            agent_html = format_agent_findings_enhanced(findings)
            return (
                plot, forecast_plot, status, metrics,
                anomalies_table, signals_table, agent_html, df, anomalies
            )
        
        def handle_evaluate(stored_df, stored_anomalies, known_anomalies_str, delay_tolerance):
            if (stored_df is None or 
                stored_anomalies is None or 
                (hasattr(stored_anomalies, 'empty') and stored_anomalies.empty)):
                return pd.DataFrame({'⚠️ 注意': ['先に異常検知分析を実行してください']}), pd.DataFrame()
            
            return run_evaluation(stored_df, stored_anomalies, known_anomalies_str, delay_tolerance)
        
        def handle_compare(stored_df, methods, thresholds_text, known_anomalies_str):
            if stored_df is None or (hasattr(stored_df, 'empty') and stored_df.empty):
                return pd.DataFrame({'⚠️ 注意': ['先に異常検知分析を実行してください']})
            
            return compare_methods(stored_df, methods, thresholds_text, known_anomalies_str)
            
            return run_evaluation(stored_df, stored_anomalies, known_anomalies_str, delay_tolerance)
        
        def handle_compare(stored_df, methods, thresholds_text, known_anomalies_str):
            if stored_df is None:
                error_df = pd.DataFrame({'⚠️ 注意': ['先に異常検知分析を実行してください']})
                return error_df
            
            return compare_methods(stored_df, methods, thresholds_text, known_anomalies_str)
        
        stored_agent_findings = gr.State(None)

        # 分析実行ボタンのイベント
        analyze_btn.click(
            fn=handle_analyze_click,
            inputs=[
                data_source, file_path, detection_method, threshold, llm_provider,
                use_web_agent, use_knowledge_agent, use_crosscheck_agent,
                use_report_agent, use_manager_agent, include_extra_indicators,
                generate_signals, forecast_days
            ],
            outputs=[
                plot_output, forecast_plot_output, status_display, metrics_display,
                anomaly_table, signals_table,
                stored_agent_findings,   # ← agent_findings をここに
                stored_df,               # ← df
                stored_anomalies         # ← anomalies
            ]
        ).then(
            fn=handle_results_display,
            inputs=[
                plot_output, forecast_plot_output, status_display, metrics_display,
                anomaly_table, signals_table,
                stored_agent_findings,   # findings
                stored_df,               # df
                stored_anomalies         # anomalies
            ],
            outputs=[
                plot_output, forecast_plot_output, status_display, metrics_display,
                anomaly_table, signals_table,
                agent_results,           # HTML へ描画
                stored_df,
                stored_anomalies
            ]
        )
        
        # 評価ボタンのイベント
        evaluate_btn.click(
            fn=handle_evaluate,
            inputs=[stored_df, stored_anomalies, known_anomalies, delay_tolerance],
            outputs=[evaluation_metrics, confusion_matrix_df]
        )
        
        # 手法比較ボタンのイベント
        compare_btn.click(
            fn=handle_compare,
            inputs=[stored_df, methods_select, thresholds_text, known_anomalies],
            outputs=comparison_results
        )
    
    return app

# メイン処理
if __name__ == "__main__":
    app = create_gradio_ui()
    app.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
