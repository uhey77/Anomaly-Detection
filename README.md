# マルチエージェントLLM異常検知フレームワーク

このプロジェクトは、大規模言語モデル（LLM）を活用したマルチエージェントアプローチによる時系列データの異常検知フレームワークです。Park (2024)の研究を基に、複数のAIエージェントが協調して金融市場データの異常を検出・分析します。

## 特徴

- 複数の専門AIエージェントによる協調分析
- 時系列データの異常検知と分析
- LLMを活用した異常の解釈と説明
- インタラクティブなWebインターフェース
- 柔軟な異常検知アルゴリズム

## エージェント構成

このフレームワークは、以下の5つの特化型エージェントで構成されています：

1. **Web情報エージェント**: 異常に関連するWeb情報を検索・分析
2. **知識ベースエージェント**: 金融市場に関するドメイン知識を提供
3. **クロスチェックエージェント**: 異常を他のデータソースと照合
4. **レポート統合エージェント**: 各エージェントの分析を統合
5. **管理者エージェント**: 最終評価と行動推奨を提供

# プロジェクト構造
詳細なプロジェクト構造については [STRUCTURE.md](https://github.com/uhey77/Anomaly-Detection/blob/main/README.md) を参照してください。

# 参考文献
- [Park, T. (2024). マルチエージェントによるLLM異常検知フレームワーク. BIS.](https://arxiv.org/html/2403.19735v1#:~:text=This%20paper%20introduces%20a%20Large,I%20analyse%20the%20S%26P%20500)
- [Alnegheimish et al. (2024). Large language models can be zero-shot anomaly detectors for time series?. MIT.](https://ar5iv.labs.arxiv.org/html/2405.14755#:~:text=detection%20task,better%20than%20large%20language%20models)
- [Zhou, Z., Yu, R. et al. (2025). Can LLMs Understand Time Series Anomalies?. UCSD.](https://openreview.net/forum?id=LGafQ1g2D2)
