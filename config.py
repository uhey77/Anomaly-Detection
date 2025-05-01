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
ANOMALY_METHOD = "z_score"  # "z_score", "iqr", または "moving_avg"
ANOMALY_THRESHOLD = 3.0  # 異常検知の閾値

# データ設定
USE_SAMPLE_DATA = True  # サンプルデータを使用するか実データをダウンロードするか
START_DATE = "1980-01-01"  # データの開始日
END_DATE = "2023-12-31"  # データの終了日

# エージェント設定
ENABLE_WEB_AGENT = True  # Web情報エージェントを有効にする
ENABLE_KNOWLEDGE_AGENT = True  # 知識ベースエージェントを有効にする
ENABLE_CROSSCHECK_AGENT = True  # クロスチェックエージェントを有効にする
ENABLE_REPORT_AGENT = True  # レポート統合エージェントを有効にする
ENABLE_MANAGER_AGENT = True  # 管理者エージェントを有効にする
