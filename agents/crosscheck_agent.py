import pandas as pd
import numpy as np
from .base_agent import BaseAgent

class CrossCheckAgent(BaseAgent):
    """異常を他のデータソースとクロスチェックするエージェント"""
    
    def __init__(self, name="クロスチェックエージェント", llm_client=None):
        """
        クロスチェックエージェントを初期化
        
        Args:
            name (str): エージェント名
            llm_client: LLMクライアントインスタンス
        """
        super().__init__(name, llm_client)
        
    def process(self, anomaly_data, context=None):
        """
        異常を他のデータソースとクロスチェック
        
        Args:
            anomaly_data (pd.DataFrame): 異常を含むデータ
            context (dict): 参照データを含む追加コンテキスト
            
        Returns:
            dict: クロスチェック結果
        """
        findings = {}
        
        # コンテキストから参照データを取得（利用可能な場合）
        reference_data = context.get('reference_data', {}) if context else {}
        
        for idx, anomaly in anomaly_data.iterrows():
            date = anomaly['Date']
            value = anomaly['Close']
            pct_change = anomaly['pct_change']
            
            # 検索用の日付フォーマット
            anomaly_date = date.strftime("%Y-%m-%d")
            
            # 参照データとのクロスチェック（簡略化）
            cross_check_results = self._cross_check_with_reference(anomaly_date, reference_data)
            
            # クロスチェック結果を分析するためのLLMクエリ
            llm_prompt = f"""
            他の市場指標とクロスチェックして、このS&P 500の異常を分析してください:
            
            日付: {anomaly_date}
            S&P 500値: {value}
            変化率: {pct_change:.2f}%
            
            他の指標とのクロスチェック:
            {cross_check_results}
            
            このクロスチェックに基づいて、この異常は:
            1. 他の指標によって確認されている（複数の市場が類似したパターンを示している）
            2. S&P 500に限定されている（他の指標は類似したパターンを示していない）
            3. 矛盾している（他の指標は反対のパターンを示している）
            
            分析し、あなたの推論を説明してください。
            """
            
            llm_analysis = self.query_llm(llm_prompt)
            
            findings[anomaly_date] = {
                "cross_check_results": cross_check_results,
                "llm_analysis": llm_analysis
            }
            
        return {
            "agent": self.name,
            "findings": findings
        }
        
    def _cross_check_with_reference(self, date, reference_data):
        """
        異常を参照データとクロスチェック
        
        Args:
            date (str): 異常の日付
            reference_data (dict): 参照データソース
            
        Returns:
            str: クロスチェック結果
        """
        # モック実装 - 実際のシステムでは実際のデータと比較
        results = []
        
        # 特定の既知の異常日のモックデータ
        if date == "1987-10-19":
            results = [
                "ダウ工業株30種平均: -22.6%（異常を確認）",
                "NASDAQ: -11.4%（異常を確認）",
                "10年国債利回り: -0.5%（市場ストレスを確認）",
                "VIX相当: 大幅上昇（異常を確認）"
            ]
        elif date == "2008-10-13":
            results = [
                "ダウ工業株30種平均: +11.1%（異常を確認）",
                "NASDAQ: +11.8%（異常を確認）",
                "10年国債利回り: +7.5%（リスクオンの動きを確認）",
                "VIX: -16.5%（信頼回復を確認）"
            ]
        elif date == "2020-03-16":
            results = [
                "ダウ工業株30種平均: -12.9%（異常を確認）",
                "NASDAQ: -12.3%（異常を確認）",
                "10年国債利回り: -24%（安全資産への逃避を確認）",
                "VIX: +25%で過去最高（極度の恐怖を確認）"
            ]
        else:
            # 他の日付に対する一般的な応答
            results = [
                "ダウ工業株30種平均: 同様のパターンが観察される",
                "NASDAQ: 同様のパターンが観察される",
                "債券市場に異常な動きなし",
                "VIX: 重要な変化なし"
            ]
            
        return "\n".join(results)
