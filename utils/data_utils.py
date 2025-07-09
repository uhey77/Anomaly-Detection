import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# realtime_data_provider.pyをインポート
try:
    from realtime_data_provider import RealTimeDataProvider
    REALTIME_AVAILABLE = True
    print("✅ リアルタイムデータプロバイダーが利用可能です")
except ImportError:
    REALTIME_AVAILABLE = False
    print("⚠️ リアルタイムデータプロバイダーが見つかりません。サンプルデータのみ利用可能です")


class DataManager:
    """データ管理クラス（リアルタイム対応）"""
    
    def __init__(self):
        if REALTIME_AVAILABLE:
            self.rt_provider = RealTimeDataProvider()
        else:
            self.rt_provider = None
    
    def load_sample_data(self, 
                        start_date=None, 
                        end_date=None,
                        symbol='sp500'):
        """サンプルデータを取得（リアルタイム対応版）"""
        
        # リアルタイムプロバイダーが利用可能な場合
        if self.rt_provider:
            try:
                # 日付の設定
                if end_date is None:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                if start_date is None:
                    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                
                # リアルタイムデータを取得
                data_dict = self.rt_provider.get_realtime_data([symbol], period="2y")
                
                if symbol in data_dict and not data_dict[symbol].empty:
                    df = data_dict[symbol].copy()
                    
                    # 日付フィルタリング
                    df['Date'] = pd.to_datetime(df['Date'])
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    df = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
                    
                    if not df.empty:
                        # データ型を確保
                        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        df = df.dropna(subset=['Close']).reset_index(drop=True)
                        print(f"✅ リアルタイムデータ取得完了: {symbol} ({len(df)}件)")
                        return df
                
            except Exception as e:
                print(f"⚠️ リアルタイムデータ取得エラー: {e}")
        
        # フォールバック: 既存のサンプルデータ生成
        print("📊 フォールバックデータを生成中...")
        return self._generate_fallback_data(start_date, end_date)
    
    def load_multi_indicator_data(self, 
                                 start_date=None,
                                 end_date=None,
                                 symbols=None):
        """複数指標のデータを取得（リアルタイム対応版）"""
        
        if symbols is None:
            symbols = ['sp500', 'vix', 'usdjpy', 'gold', 'nasdaq']
        
        if self.rt_provider:
            try:
                # リアルタイムデータを取得
                data_dict = self.rt_provider.get_realtime_data(symbols, period="2y")
                
                # 日付でフィルタリング
                if start_date and end_date:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    for symbol in data_dict:
                        df = data_dict[symbol]
                        df['Date'] = pd.to_datetime(df['Date'])
                        data_dict[symbol] = df[(df['Date'] >= start_dt) & (df['Date'] <= end_dt)]
                
                # 既存形式との互換性のため名前をマッピング
                result = {}
                
                # メインデータ（S&P500）
                if 'sp500' in data_dict and not data_dict['sp500'].empty:
                    result['sp500'] = data_dict['sp500']
                    
                    # 出来高データ
                    volume_df = data_dict['sp500'][['Date', 'Volume']].copy()
                    result['volume'] = volume_df
                
                # VIXデータ
                if 'vix' in data_dict and not data_dict['vix'].empty:
                    vix_df = data_dict['vix'][['Date', 'Close']].copy()
                    vix_df = vix_df.rename(columns={'Close': 'VIX'})
                    result['vix'] = vix_df
                
                # USD/JPYデータ
                if 'usdjpy' in data_dict and not data_dict['usdjpy'].empty:
                    usdjpy_df = data_dict['usdjpy'][['Date', 'Close']].copy()
                    usdjpy_df = usdjpy_df.rename(columns={'Close': 'USDJPY'})
                    result['usdjpy'] = usdjpy_df
                
                # その他のデータ
                for symbol in ['gold', 'nasdaq']:
                    if symbol in data_dict and not data_dict[symbol].empty:
                        result[symbol] = data_dict[symbol]
                
                print(f"✅ 複数指標データ取得完了: {len(result)}個の指標")
                return result
                
            except Exception as e:
                print(f"⚠️ 複数指標データ取得エラー: {e}")
        
        # フォールバック: 既存の関数を呼び出し
        return load_multi_indicator_data_original(start_date, end_date)
    
    def _generate_fallback_data(self, start_date=None, end_date=None):
        """フォールバック用のサンプルデータ生成"""
        print("📈 サンプルデータを生成中...")
        
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = pd.to_datetime(end_date)
            
        if start_date is None:
            start_date = end_date - timedelta(days=730)
        else:
            start_date = pd.to_datetime(start_date)
        
        # 日付範囲を生成
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # S&P500のような価格データを模擬
        np.random.seed(42)
        n_days = len(dates)
        
        # 初期価格
        initial_price = 4000.0
        
        # ランダムウォーク + トレンド
        returns = np.random.normal(0.0008, 0.02, n_days)
        returns[0] = 0
        
        prices = [initial_price]
        for i in range(1, n_days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 1.0))
        
        # OHLC価格を生成
        closes = np.array(prices)
        opens = np.concatenate([[closes[0]], closes[:-1]])
        highs = closes * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
        volumes = np.random.randint(1000000, 10000000, n_days)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes,
            'Volume': volumes
        })
        
        return df


def load_sample_data_original(start_date="1980-01-01", end_date="2023-12-31"):
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

def load_sample_data(start_date=None, end_date=None):
    """既存関数の互換性維持（リアルタイム対応）"""
    manager = DataManager()
    return manager.load_sample_data(start_date, end_date)

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

def load_multi_indicator_data_original(start_date="1980-01-01", end_date="2023-12-31", include_mock_indicators=True):
    """
    複数の指標を含むデータを生成
    
    Args:
        start_date (str): 開始日
        end_date (str): 終了日
        include_mock_indicators (bool): モック指標を含めるかどうか
        
    Returns:
        dict: 指標データの辞書
    """
    # 基本のS&P 500データを読み込む
    sp500_df = load_sample_data(start_date, end_date)
    
    # 複数指標のデータ辞書を初期化
    data_dict = {
        'sp500': sp500_df
    }
    
    if include_mock_indicators:
        dates = sp500_df['Date']
        n = len(dates)
        np.random.seed(42)
        
        # 1. 出来高データの生成（基本トレンド + ノイズ）
        base_volume = 1000000
        volume_trend = np.linspace(0, 2, n)  # 時間とともに増加
        volume_noise = np.random.normal(0, 0.2, n)
        volumes = base_volume * (1 + volume_trend + volume_noise)
        
        # 特定の日に異常な出来高を設定
        # 1987年10月19日（ブラックマンデー）
        black_monday = pd.Timestamp('1987-10-19')
        if black_monday in dates:
            idx = dates.get_loc(black_monday)
            volumes[idx] = volumes[idx-1] * 4.5  # 大量出来高
        
        # 2008年10月13日の急騰
        oct_2008_surge = pd.Timestamp('2008-10-13')
        if oct_2008_surge in dates:
            idx = dates.get_loc(oct_2008_surge)
            volumes[idx] = volumes[idx-1] * 3.1  # 大量出来高
        
        # 2020年3月16日のCOVID-19による下落
        covid_crash = pd.Timestamp('2020-03-16')
        if covid_crash in dates:
            idx = dates.get_loc(covid_crash)
            volumes[idx] = volumes[idx-1] * 3.4  # 大量出来高
        
        # 出来高データフレームを作成
        volume_df = pd.DataFrame({
            'Date': dates,
            'Volume': volumes.astype(int)
        })
        
        # 2. VIXデータの生成（変動性 + 逆相関）
        base_vix = 20
        sp500_returns = sp500_df['Close'].pct_change().fillna(0)
        vix_values = base_vix - sp500_returns * 100 + np.random.normal(0, 2, n)
        vix_values = np.maximum(vix_values, 9)  # VIXの最小値を9に設定
        
        # 特定の日に異常なVIX値を設定
        if black_monday in dates:
            idx = dates.get_loc(black_monday)
            vix_values[idx] = 80  # 極度の恐怖
        
        if oct_2008_surge in dates:
            idx = dates.get_loc(oct_2008_surge)
            vix_values[idx] = 50  # 高い不確実性の中での反発
        
        if covid_crash in dates:
            idx = dates.get_loc(covid_crash)
            vix_values[idx] = 85  # パンデミックによる極度の恐怖
        
        # VIXデータフレームを作成
        vix_df = pd.DataFrame({
            'Date': dates,
            'VIX': vix_values
        })
        
        # 3. ドル円データの生成
        base_usdjpy = 110
        usdjpy_trend = np.random.normal(0, 0.002, n).cumsum()  # ランダムウォーク
        usdjpy_values = base_usdjpy * (1 + usdjpy_trend)
        
        # 特定の日に異常なドル円値を設定
        if black_monday in dates:
            idx = dates.get_loc(black_monday)
            usdjpy_values[idx] = usdjpy_values[idx-1] * 0.97  # 円高（安全通貨への逃避）
        
        if oct_2008_surge in dates:
            idx = dates.get_loc(oct_2008_surge)
            usdjpy_values[idx] = usdjpy_values[idx-1] * 1.02  # 円安（リスクオン）
        
        if covid_crash in dates:
            idx = dates.get_loc(covid_crash)
            usdjpy_values[idx] = usdjpy_values[idx-1] * 0.972  # 円高（安全通貨への逃避）
        
        # ドル円データフレームを作成
        usdjpy_df = pd.DataFrame({
            'Date': dates,
            'USDJPY': usdjpy_values
        })
        
        # データ辞書に追加
        data_dict['volume'] = volume_df
        data_dict['vix'] = vix_df
        data_dict['usdjpy'] = usdjpy_df
    
    return data_dict

def load_multi_indicator_data(start_date=None, end_date=None):
    """既存関数の互換性維持（リアルタイム対応）"""
    manager = DataManager()
    return manager.load_multi_indicator_data(start_date, end_date)