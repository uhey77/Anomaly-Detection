import logging
import time
from datetime import datetime, timedelta
from typing import ClassVar

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class RealTimeDataProvider:
    """リアルタイム金融データ提供クラス"""

    # 主要な銘柄・指数の定義
    SYMBOLS: ClassVar[dict[str, str]] = {
        # 主要株価指数
        "sp500": "^GSPC",  # S&P 500
        "nasdaq": "^IXIC",  # NASDAQ
        "dow": "^DJI",  # ダウ平均
        "russell2000": "^RUT",  # Russell 2000
        "nikkei": "^N225",  # 日経平均
        "ftse": "^FTSE",  # FTSE 100
        "dax": "^GDAXI",  # DAX
        "cac40": "^FCHI",  # CAC 40
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
        "gold": "GC=F",  # 金先物
        "silver": "SI=F",  # 銀先物
        "oil_wti": "CL=F",  # WTI原油
        "oil_brent": "BZ=F",  # ブレント原油
        "natural_gas": "NG=F",  # 天然ガス
        "copper": "HG=F",  # 銅
        # 通貨
        "usdjpy": "JPY=X",  # ドル円
        "eurusd": "EURUSD=X",  # ユーロドル
        "gbpusd": "GBPUSD=X",  # ポンドドル
        "usdcad": "USDCAD=X",  # ドルカナダ
        "audusd": "AUDUSD=X",  # オーストラリアドル
        "usdjpy_inverted": "JPYUSD=X",  # 円ドル
        # 債券
        "us_10y": "^TNX",  # 10年米国債
        "us_30y": "^TYX",  # 30年米国債
        "us_2y": "^IRX",  # 2年米国債
        "de_10y": "^TNX",  # ドイツ10年債（代替）
        # 暗号通貨
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        # セクターETF
        "tech_etf": "XLK",  # テクノロジー
        "finance_etf": "XLF",  # 金融
        "energy_etf": "XLE",  # エネルギー
        "healthcare_etf": "XLV",  # ヘルスケア
        "consumer_etf": "XLY",  # 消費財
        # 恐怖指数・ボラティリティ
        "vix": "^VIX",  # VIX恐怖指数
        "vxn": "^VXN",  # NASDAQ VIX
        "rvx": "^RVX",  # Russell 2000 VIX
    }

    # デフォルトで取得する主要シンボル
    DEFAULT_SYMBOLS: ClassVar[tuple[str, ...]] = (
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
    )

    def __init__(self, cache_duration_minutes: int = 5) -> None:
        """
        Args:
            cache_duration_minutes: データキャッシュの有効期間（分）
        """
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self.data_cache: dict[str, pd.DataFrame] = {}
        self.last_update: dict[str, datetime] = {}

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

        except Exception:  # yfinance由来のさまざまな入力形式を扱う
            logger.exception("技術指標の計算に失敗しました")

        return df

    # ------------------------------------------------------------------ #
    # パブリック API
    # ------------------------------------------------------------------ #
    def get_realtime_data(
        self,
        symbols: list[str] | None = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        リアルタイムデータを取得

        Args:
            symbols: 取得するシンボルリスト（None の場合はデフォルト）
            period: データ期間
            interval: データ間隔

        Returns:
            Dict[str, pd.DataFrame]: シンボル名をキーとするデータフレーム辞書
        """
        requested_symbols = symbols or self.DEFAULT_SYMBOLS
        data_dict: dict[str, pd.DataFrame] = {}

        for symbol_name in requested_symbols:
            if symbol_name not in self.SYMBOLS:
                logger.warning("未知のシンボル '%s' をスキップします", symbol_name)
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
                    logger.warning("%s のデータが取得できませんでした", symbol_name)
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
                df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

                # NaN 除去
                df = df.dropna(subset=["Close"])
                if df.empty:
                    logger.warning("%s の有効なデータがありません", symbol_name)
                    continue

                # 指標付与
                df = self._add_technical_indicators(df)

                # キャッシュ
                self.data_cache[symbol_name] = df
                self.last_update[symbol_name] = datetime.now()
                data_dict[symbol_name] = df

                time.sleep(0.1)  # API 制限対策

            except Exception:  # yfinanceの通信・解析例外をシンボル単位で隔離する
                logger.exception("%s のデータ取得に失敗しました", symbol_name)

        return data_dict

    def get_market_summary(self) -> dict[str, dict]:
        """市場サマリーを取得"""
        try:
            major_indices = ["sp500", "nasdaq", "dow", "nikkei", "vix"]
            data = self.get_realtime_data(major_indices, period="5d", interval="1d")

            summary: dict[str, dict] = {}
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

        except Exception:  # yfinance由来の例外は空のサマリーにフォールバックする
            logger.exception("市場サマリーの取得に失敗しました")
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

        except Exception:  # yfinance由来の例外は空データにフォールバックする
            logger.exception("イントラデイデータの取得に失敗しました: %s", symbol)
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 便利ユーティリティ
    # ------------------------------------------------------------------ #
    def get_available_symbols(self) -> dict[str, str]:
        """利用可能なシンボル一覧を取得"""
        return self.SYMBOLS.copy()

    def validate_market_hours(self) -> dict[str, bool]:
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
