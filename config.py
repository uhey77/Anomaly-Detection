"""マルチエージェントLLM異常検知フレームワークの設定"""
import os
from dotenv import load_dotenv

# .envファイルがあればロード（APIキーなどの機密情報用）
load_dotenv()

# LLM設定（非機密情報）
LLM_PROVIDER = "mock"  # "openai", "huggingface", または "mock"
OPENAI_MODEL = "gpt-4.1-mini"  # OpenAIで使用するモデル
HF_MODEL = "mistral/mistral-7b-instruct-v0.2"  # Hugging Faceで使用するモデル

# 機密情報（.envから読み込み）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # OpenAI APIキー
HF_API_KEY = os.getenv("HF_API_KEY", "")  # Hugging Face APIキー

# 異常検知設定
ANOMALY_METHOD = "z_score"  # "z_score", "iqr", "moving_avg", "isolation_forest", "deep_svdd"
ANOMALY_THRESHOLD = 3.0  # 異常検知の閾値

# 追加の異常検知パラメータ
ANOMALY_PARAMS = {
    # Isolation Forest用パラメータ
    "isolation_forest": {
        "contamination": 0.05,  # 異常の予想割合
        "n_estimators": 100,    # 決定木の数
        "random_state": 42      # 乱数シード
    },
    # Deep SVDD用パラメータ
    "deep_svdd": {
        "threshold": 0.95,      # 異常判定の閾値（百分位数ベース）
        "epochs": 50,           # トレーニングのエポック数
        "batch_size": 32,       # バッチサイズ
        "random_state": 42      # 乱数シード
    },
    # 移動平均用パラメータ
    "moving_avg": {
        "window": 20            # 移動平均の窓サイズ
    }
}

# TimesFM/LSTM予測モデル設定
FORECASTING_MODEL = "lstm"  # "lstm" または "timesfm"
FORECASTING_PARAMS = {
    "lstm": {
        "units": 50,            # LSTMユニット数
        "dropout": 0.2,         # ドロップアウト率
        "lookback": 60,         # 遡る期間
        "epochs": 50,           # トレーニングのエポック数
        "batch_size": 32        # バッチサイズ
    },
    "timesfm": {
        "d_model": 64,          # モデルの次元
        "num_heads": 4,         # Attention Headsの数
        "num_layers": 2,        # Transformer Layersの数
        "dropout": 0.1,         # ドロップアウト率
        "lookback": 60,         # 遡る期間
        "epochs": 50,           # トレーニングのエポック数
        "batch_size": 32        # バッチサイズ
    }
}

# シグナル生成設定
SIGNAL_THRESHOLD = 0.5  # シグナル生成の閾値
SIGNAL_WINDOW = 5       # 価格トレンド判定の窓サイズ

# データ設定
USE_SAMPLE_DATA = True  # サンプルデータを使用するか実データをダウンロードするか
START_DATE = "1980-01-01"  # データの開始日
END_DATE = "2023-12-31"  # データの終了日
INCLUDE_EXTRA_INDICATORS = True  # 追加指標（出来高、VIX、ドル円）を含めるか

# エージェント設定
ENABLE_WEB_AGENT = True  # Web情報エージェントを有効にする
ENABLE_KNOWLEDGE_AGENT = True  # 知識ベースエージェントを有効にする
ENABLE_CROSSCHECK_AGENT = True  # クロスチェックエージェントを有効にする
ENABLE_REPORT_AGENT = True  # レポート統合エージェントを有効にする
ENABLE_MANAGER_AGENT = True  # 管理者エージェントを有効にする


# =============================================================================
# リアルタイムデータ設定（追加）
# =============================================================================

# Yahoo Finance設定
YFINANCE_CONFIG = {
    'timeout': 30,
    'retry_count': 3,
    'retry_delay': 1,
    'cache_duration': 300,  # 5分
}

# サポートする期間とインターバル
SUPPORTED_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
SUPPORTED_INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

# リアルタイム対応でANOMALY_PARAMSの一部を調整（既存があれば上書き）
ANOMALY_PARAMS.update({
    'isolation_forest': {
        'contamination': 0.1,
        'n_estimators': 50,  # 高速化のため削減
        'random_state': 42,
        'max_samples': 'auto',
        'max_features': 1.0
    },
    'deep_svdd': {
        'threshold': 0.95,
        'epochs': 20,  # 高速化のため削減
        'batch_size': 16,
        'learning_rate': 0.001,
        'random_state': 42
    }
})

# キャッシュ設定
CACHE_CONFIG = {
    'enable_cache': True,
    'cache_dir': './cache',
    'cache_ttl': 300,  # 5分
}

# リアルタイムUI設定
REALTIME_UI_CONFIG = {
    'auto_refresh': False,
    'refresh_interval': 60,  # 秒
    'max_symbols': 20,
    'enable_alerts': False,
}