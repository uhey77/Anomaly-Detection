from .base_agent import BaseAgent

class KnowledgeBaseAgent(BaseAgent):
    """金融市場に関するドメイン知識を提供するエージェント"""
    
    def __init__(self, name="知識ベースエージェント", llm_client=None):
        """
        知識ベースエージェントを初期化
        
        Args:
            name (str): エージェント名
            llm_client: LLMクライアントインスタンス
        """
        super().__init__(name, llm_client)
        
    def process(self, anomaly_data, context=None):
        """
        異常に関するドメイン知識を提供
        
        Args:
            anomaly_data (pd.DataFrame): 異常を含むデータ
            context (dict, optional): 追加コンテキスト
            
        Returns:
            dict: 知識ベースの結果
        """
        findings = {}
        
        for idx, anomaly in anomaly_data.iterrows():
            date = anomaly['Date']
            value = anomaly['Close']
            pct_change = anomaly['pct_change']
            
            # 検索用の日付フォーマット
            anomaly_date = date.strftime("%Y-%m-%d")
            
            # 金融知識に関するLLMクエリ
            llm_prompt = f"""
            金融市場の専門家として、S&P 500指数のこの異常を分析してください:
            
            日付: {anomaly_date}
            S&P 500値: {value}
            変化率: {pct_change:.2f}%
            
            以下を提供してください:
            1. 歴史的コンテキスト: この日付またはその近くに何か知られている主要な市場イベントはありますか？
            2. 典型的な市場行動: このタイプの動き（方向と大きさ）は一般的ですか、それとも稀ですか？
            3. 潜在的原因: このような市場の動きを説明する典型的な要因は何ですか？
            4. 類似の歴史的先例: 市場の歴史で比較可能なイベントはありますか？
            
            金融市場の知識に基づいて分析してください。
            """
            
            llm_analysis = self.query_llm(llm_prompt)
            
            findings[anomaly_date] = {
                "llm_analysis": llm_analysis
            }
            
        return {
            "agent": self.name,
            "findings": findings
        }
