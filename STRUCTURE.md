# プロジェクト構造

## ディレクトリとファイルの説明

anomaly_detection/
├── data/                        # データ格納ディレクトリ
│   └── sp500_sample.csv         # サンプルデータ
├── agents/                      # エージェントモジュール
│   ├── init.py              # パッケージ初期化
│   ├── base_agent.py            # 基本エージェントクラス
│   ├── web_agent.py             # Web情報エージェント
│   ├── knowledge_agent.py       # 知識ベースエージェント
│   ├── crosscheck_agent.py      # クロスチェックエージェント
│   ├── report_agent.py          # レポート統合エージェント
│   └── manager_agent.py         # 管理者エージェント
├── detection/                   # 異常検知モジュール
│   ├── init.py              # パッケージ初期化
│   └── anomaly_detector.py      # 異常検知アルゴリズム
├── utils/                       # ユーティリティモジュール
│   ├── init.py              # パッケージ初期化
│   ├── data_utils.py            # データ処理ユーティリティ
│   └── llm_clients.py           # LLMクライアント
├── app.py                       # Webアプリケーション (Gradio)
├── main.py                      # コマンドライン実行スクリプト
├── config.py                    # 設定ファイル
├── requirements.txt             # 依存パッケージリスト
├── .env.example                 # 環境変数テンプレート
└── .gitignore                   # Gitで無視するファイルリスト


## 主要コンポーネントの説明

### エージェントモジュール

- **base_agent.py**: すべてのエージェントの基本クラスを定義。LLMクエリ機能を提供します。
- **web_agent.py**: 異常日に関する情報をWeb（またはモック）から検索し分析します。
- **knowledge_agent.py**: 金融市場の知識を活用して異常を分析します。
- **crosscheck_agent.py**: 異常を他の市場指標と照合します。
- **report_agent.py**: 他のエージェントからの分析を統合した総合レポートを作成します。
- **manager_agent.py**: レポートを経営的視点でレビューし、最終評価と推奨事項を提供します。


### 異常検知モジュール (`detection/`)

- **anomaly_detector.py**: 複数の異常検知アルゴリズム（Z-score、IQR、移動平均）を実装しています。


### ユーティリティモジュール (`utils/`)

- **data_utils.py**: データの読み込み、保存、可視化のためのユーティリティ関数を提供します。
- **llm_clients.py**: OpenAI、HuggingFace、およびモック用のLLMクライアントを実装しています。


### アプリケーション

- **app.py**: Gradioを使用したインタラクティブなWebインターフェースを提供します。
- **main.py**: コマンドラインから実行するためのスクリプトです。