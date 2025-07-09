import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


class RealTimeDataProvider:
    """リアルタイム金融データ提供クラス"""

    # 主要な銘柄・指数の定義
    SYMBOLS = {
        # 主要株価指数
        "sp500": "^GSPC",       # S&P 500
        "nasdaq": "^IXIC",      # NASDAQ
        "dow": "^DJI",          # ダウ平均
        "russell2000": "^RUT",  # Russell 2000
        "nikkei": "^N225",      # 日経平均
        "ftse": "^FTSE",        # FTSE 100
        "dax": "^GDAXI",        # DAX
        "cac40": "^FCHI",       # CAC 40

        # 主要個別株
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "meta": "META",
        "berkshire": "BRK-A",
        "toyota": "TM",
        "asml": "ASML",

        # 商品
        "gold": "GC=F",          # 金先物
        "silver": "SI=F",        # 銀先物
        "oil_wti": "CL=F",       # WTI原油
        "oil_brent": "BZ=F",     # ブレント原油
        "natural_gas": "NG=F",   # 天然ガス
        "copper": "HG=F",        # 銅

        # 通貨
        "usdjpy": "JPY=X",       # ドル円
        "eurusd": "EURUSD=X",    # ユーロドル
        "gbpusd": "GBPUSD=X",    # ポンドドル
        "usdcad": "USDCAD=X",    # ドルカナダ
        "audusd": "AUDUSD=X",    # オーストラリアドル
        "usdjpy_inverted": "JPYUSD=X",  # 円ドル

        # 債券
        "us_10y": "^TNX",        # 10年米国債
        "us_30y": "^TYX",        # 30年米国債
        "us_2y": "^IRX",         # 2年米国債
        "de_10y": "^TNX",        # ドイツ10年債（代替）

        # 暗号通貨
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",

        # セクターETF
        "tech_etf": "XLK",       # テクノロジー
        "finance_etf": "XLF",    # 金融
        "energy_etf": "XLE",     # エネルギー
        "healthcare_etf": "XLV", # ヘルスケア
        "consumer_etf": "XLY",   # 消費財

        # 恐怖指数・ボラティリティ
        "vix": "^VIX",           # VIX恐怖指数
        "vxn": "^VXN",           # NASDAQ VIX
        "rvx": "^RVX",           # Russell 2000 VIX
    }

    # デフォルトで取得する主要シンボル
    DEFAULT_SYMBOLS = [
        "sp500",
        "nasdaq",
        "dow",
        "nikkei",
        "apple",
        "microsoft",
        "tesla",
        "nvidia",
        "gold",
        "oil_wti",
        "usdjpy",
        "eurusd",
        "bitcoin",
        "vix",
        "us_10y",
    ]

    def __init__(self, cache_duration_minutes: int = 5) -> None:
        """
        Args:
            cache_duration_minutes: データキャッシュの有効期間（分）
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.last_update: Dict[str, datetime] = {}

    # ------------------------------------------------------------------ #
    # 内部ユーティリティ
    # ------------------------------------------------------------------ #
    def _is_cache_valid(self, symbol: str) -> bool:
        """キャッシュが有効かどうかをチェック"""
        return (
            symbol in self.last_update
            and datetime.now() - self.last_update[symbol] < self.cache_duration
        )

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な技術指標を追加"""
        df = df.copy()

        try:
            # 移動平均
            df["MA_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
            df["MA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
            df["MA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

            # ボリンジャーバンド
            rolling_mean = df["Close"].rolling(window=20, min_periods=1).mean()
            rolling_std = df["Close"].rolling(window=20, min_periods=1).std()
            df["BB_Upper"] = rolling_mean + rolling_std * 2
            df["BB_Lower"] = rolling_mean - rolling_std * 2

            # RSI（簡易版）
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-8)
            df["RSI"] = 100 - 100 / (1 + rs)

            # 変化率
            df["pct_change"] = df["Close"].pct_change() * 100
            df["pct_change_abs"] = df["pct_change"].abs()

            # ボラティリティ（20日）
            df["volatility"] = df["Close"].rolling(window=20, min_periods=1).std()

            # 出来高比率
            volume_ma = df["Volume"].rolling(window=20, min_periods=1).mean()
            df["volume_ratio"] = df["Volume"] / (volume_ma + 1e-8)

        except Exception as e:  # noqa: BLE001
            print(f"技術指標の計算でエラーが発生しました: {e}")

        return df

    # ------------------------------------------------------------------ #
    # パブリック API
    # ------------------------------------------------------------------ #
    def get_realtime_data(
        self,
        symbols: Optional[List[str]] = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        リアルタイムデータを取得

        Args:
            symbols: 取得するシンボルリスト（None の場合はデフォルト）
            period: データ期間
            interval: データ間隔

        Returns:
            Dict[str, pd.DataFrame]: シンボル名をキーとするデータフレーム辞書
        """
        symbols = symbols or self.DEFAULT_SYMBOLS
        data_dict: Dict[str, pd.DataFrame] = {}

        for symbol_name in symbols:
            if symbol_name not in self.SYMBOLS:
                print(f"警告: 未知のシンボル '{symbol_name}' をスキップします")
                continue

            ticker_symbol = self.SYMBOLS[symbol_name]

            try:
                # キャッシュ利用可否
                if self._is_cache_valid(symbol_name):
                    data_dict[symbol_name] = self.data_cache[symbol_name]
                    continue

                # データ取得
                ticker = yf.Ticker(ticker_symbol)
                hist_data = ticker.history(period=period, interval=interval)

                if hist_data.empty:
                    print(f"警告: {symbol_name} のデータが取得できませんでした")
                    continue

                df = hist_data.reset_index()

                # 日付カラム統一
                if "Datetime" in df.columns:
                    df["Date"] = df["Datetime"]
                elif "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                else:
                    df["Date"] = df.index

                # カラム補完
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in df.columns:
                        df[col] = 0 if col == "Volume" else df["Close"]

                # 型変換
                numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
                df[numeric_columns] = df[numeric_columns].apply(
                    pd.to_numeric, errors="coerce"
                )

                # NaN 除去
                df = df.dropna(subset=["Close"])
                if df.empty:
                    print(f"警告: {symbol_name} の有効なデータがありません")
                    continue

                # 指標付与
                df = self._add_technical_indicators(df)

                # キャッシュ
                self.data_cache[symbol_name] = df
                self.last_update[symbol_name] = datetime.now()
                data_dict[symbol_name] = df

                time.sleep(0.1)  # API 制限対策

            except Exception as e:  # noqa: BLE001
                print(f"エラー: {symbol_name} のデータ取得中にエラーが発生しました: {e}")

        return data_dict

    def get_market_summary(self) -> Dict[str, Dict]:
        """市場サマリーを取得"""
        try:
            major_indices = ["sp500", "nasdaq", "dow", "nikkei", "vix"]
            data = self.get_realtime_data(major_indices, period="5d", interval="1d")

            summary: Dict[str, Dict] = {}
            for symbol, df in data.items():
                if df.empty:
                    continue

                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest

                change = latest["Close"] - prev["Close"]
                change_pct = (change / prev["Close"]) * 100 if prev["Close"] != 0 else 0

                summary[symbol] = {
                    "current_price": float(latest["Close"]),
                    "change": float(change),
                    "change_pct": float(change_pct),
                    "volume": float(latest["Volume"]),
                    "high_52w": float(df["Close"].max()),
                    "low_52w": float(df["Close"].min()),
                    "last_update": latest["Date"].strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(latest["Date"], "strftime")
                    else str(latest["Date"]),
                }

            return summary

        except Exception as e:  # noqa: BLE001
            print(f"市場サマリー取得エラー: {e}")
            return {}

    def get_intraday_data(
        self,
        symbol: str,
        interval: str = "1m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        イントラデイ（分足）データを取得
        """
        if symbol not in self.SYMBOLS:
            raise ValueError(f"未知のシンボル: {symbol}")

        ticker_symbol = self.SYMBOLS[symbol]

        try:
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                return pd.DataFrame()

            df = data.reset_index()
            if "Datetime" in df.columns:
                df["Date"] = df["Datetime"]

            return self._add_technical_indicators(df)

        except Exception as e:  # noqa: BLE001
            print(f"イントラデイデータ取得エラー ({symbol}): {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 便利ユーティリティ
    # ------------------------------------------------------------------ #
    def get_available_symbols(self) -> Dict[str, str]:
        """利用可能なシンボル一覧を取得"""
        return self.SYMBOLS.copy()

    def validate_market_hours(self) -> Dict[str, bool]:
        """主要市場の取引時間をチェック（簡易版）"""
        now = datetime.now()

        return {
            "US": self._is_us_market_open(now),
            "Japan": self._is_japan_market_open(now),
            "Europe": self._is_europe_market_open(now),
            "Crypto": True,  # 24/7
        }

    def _is_us_market_open(self, dt: datetime) -> bool:
        """米国市場の開場状況（平日のみ・祝日判定なし）"""
        return dt.weekday() < 5

    def _is_japan_market_open(self, dt: datetime) -> bool:
        """日本市場の開場状況（平日のみ・祝日判定なし）"""
        return dt.weekday() < 5

    def _is_europe_market_open(self, dt: datetime) -> bool:
        """欧州市場の開場状況（平日のみ・祝日判定なし）"""
        return dt.weekday() < 5


# ---------------------------------------------------------------------- #
# テストスクリプト
# ---------------------------------------------------------------------- #
def test_realtime_data_provider() -> None:
    provider = RealTimeDataProvider()

    print("=== リアルタイムデータプロバイダーテスト ===")

    # 1. 基本データ取得
    print("\n1. 主要指数データ取得テスト")
    data = provider.get_realtime_data(["sp500", "nasdaq", "vix"], period="1mo")
    for symbol, df in data.items():
        if not df.empty:
            latest = df.iloc[-1]
            print(f"{symbol}: 最新価格 {latest['Close']:.2f}, 日付 {latest['Date']}")

    # 2. 市場サマリー
    print("\n2. 市場サマリーテスト")
    summary = provider.get_market_summary()
    for symbol, info in summary.items():
        print(f"{symbol}: {info['current_price']:.2f} ({info['change_pct']:+.2f}%)")

    # 3. 利用可能シンボル
    print(f"\n3. 利用可能シンボル数: {len(provider.get_available_symbols())}")

    # 4. 市場開場状況
    print("\n4. 市場開場状況")
    market_status = provider.validate_market_hours()
    for market, is_open in market_status.items():
        status = "開場中" if is_open else "閉場中"
        print(f"{market}: {status}")


if __name__ == "__main__":
    test_realtime_data_provider()
