# マルチエージェントLLM異常検知フレームワーク

金融時系列データの異常を検出し、複数のLLMエージェントで原因分析・評価・将来予測を行うPythonアプリケーションです。

## 主な機能

- Z-Score、IQR、移動平均、Isolation Forest、Deep SVDDによる異常検知
- Web情報・知識・クロスチェック・レポート・管理者エージェントによる協調分析
- 売買シグナル生成とLSTMベースの価格予測
- 既知の異常日を使った精度評価と手法比較
- GradioによるWeb UIとCLIの2つの実行方法

## 必要環境

- Python 3.12
- uv（推奨）またはpip

## セットアップ

ロック済みの本番依存をインストールします。

~~~bash
uv venv --python 3.12
uv pip install -r requirements.lock
~~~

開発・検証を行う場合は、代わりに `requirements-dev.lock` を使用します。

APIを利用する場合は、実行環境に次の環境変数を設定してください。

~~~bash
export OPENAI_API_KEY="..."
export HF_API_KEY="..."
~~~

APIキーを設定しない場合でも、Mockプロバイダーで主要な画面と分析フローを確認できます。

## 実行

Web UI:

~~~bash
uv run python app_improved.py
~~~

CLI:

~~~bash
uv run python main.py
~~~

Web UIは既定で `http://localhost:7861` に起動します。

## データ形式

アップロードするCSVまたはXLSXには、日付と価格を表す列が必要です。推奨する列名は次のとおりです。

- `Date`: 日付
- `Close`: 終値
- `Volume`: 出来高（任意）
- `VIX`: VIX指数（任意）
- `USDJPY`: ドル円（任意）

`Date`または`Close`がない場合は、先頭2列を日付と価格として扱います。

## プロジェクト構成

~~~text
.
├── agents/                    # LLMエージェント
├── data/                      # サンプルデータ
├── detection/                 # 異常検知器と生成ファクトリー
├── models/                    # 時系列予測モデル
├── services/                  # データ読み込みなどのアプリケーションサービス
├── ui/                        # Gradio UIのスタイル
├── utils/                     # LLMクライアントとシグナル生成
├── app_improved.py            # 正式なWeb UI
├── main.py                    # CLI
├── config.py                  # 検出・予測・エージェント設定
├── evaluation.py              # 精度評価
└── realtime_data_provider.py  # Yahoo Financeデータ取得
~~~

異常検知器は `detection.create_detector()`、売買シグナルは
`utils.signal_generator.SignalGenerator` を唯一の実装元として利用します。

## 検証

~~~bash
ruff check .
ruff format --check .
python -m unittest discover -v
~~~

## 参考文献

- [Park (2024), マルチエージェントによるLLM異常検知フレームワーク](https://arxiv.org/html/2403.19735v1)
- [Alnegheimish et al. (2024), Large language models can be zero-shot anomaly detectors for time series?](https://ar5iv.labs.arxiv.org/html/2405.14755)
- [Zhou et al. (2025), Can LLMs Understand Time Series Anomalies?](https://openreview.net/forum?id=LGafQ1g2D2)
