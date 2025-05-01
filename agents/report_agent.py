from .base_agent import BaseAgent

class ReportIntegrationAgent(BaseAgent):
    """他のエージェントからの結果を包括的なレポートに統合するエージェント"""
    
    def __init__(self, name="レポート統合エージェント", llm_client=None):
        """
        レポート統合エージェントを初期化
        
        Args:
            name (str): エージェント名
            llm_client: LLMクライアントインスタンス
        """
        super().__init__(name, llm_client)
        
    def process(self, anomaly_data, context=None):
        """
        他のエージェントからの結果を統合
        
        Args:
            anomaly_data (pd.DataFrame): 異常を含むデータ
            context (dict): エージェント結果を含む追加コンテキスト
            
        Returns:
            dict: 統合レポート
        """
        if not context or 'agent_findings' not in context:
            return {
                "agent": self.name,
                "error": "コンテキストにエージェント結果が提供されていません"
            }
            
        agent_findings = context['agent_findings']
        
        # 各異常の統合レポートを準備
        integrated_reports = {}
        
        for idx, anomaly in anomaly_data.iterrows():
            date = anomaly['Date']
            value = anomaly['Close']
            pct_change = anomaly['pct_change']
            
            # 検索用の日付フォーマット
            anomaly_date = date.strftime("%Y-%m-%d")
            
            # この日付のすべてのエージェントからの結果を収集
            date_findings = {}
            for agent_name, agent_result in agent_findings.items():
                if agent_name != self.name:  # 自分自身をスキップ
                    agent_date_findings = agent_result.get('findings', {}).get(anomaly_date, {})
                    date_findings[agent_name] = agent_date_findings
            
            # LLM統合用のデータを準備
            web_analysis = date_findings.get('Web情報エージェント', {}).get('llm_analysis', 'Web情報なし')
            knowledge_analysis = date_findings.get('知識ベースエージェント', {}).get('llm_analysis', '知識ベース情報なし')
            crosscheck_analysis = date_findings.get('クロスチェックエージェント', {}).get('llm_analysis', 'クロスチェック情報なし')
            
            # 結果を統合するためのLLMクエリ
            llm_prompt = f"""
            株式市場の異常に関する以下の分析を包括的なレポートに統合してください:
            
            異常詳細:
            日付: {anomaly_date}
            S&P 500値: {value}
            変化率: {pct_change:.2f}%
            
            WEB情報分析:
            {web_analysis}
            
            知識ベース分析:
            {knowledge_analysis}
            
            クロスチェック分析:
            {crosscheck_analysis}
            
            以下を含む包括的な統合レポートを提供してください:
            1. 異常とその重要性の要約
            2. 利用可能なすべての情報に基づく原因の説明
            3. 説明に対する信頼度評価（高、中、低）
            4. 異なる情報源間の不一致の強調
            5. 必要に応じてさらなる調査のための推奨事項
            
            レポートは簡潔で、事実に基づき、プロフェッショナルに書かれている必要があります。
            """
            
            integrated_report = self.query_llm(llm_prompt)
            
            integrated_reports[anomaly_date] = {
                "anomaly_details": {
                    "date": anomaly_date,
                    "value": float(value),
                    "pct_change": float(pct_change)
                },
                "individual_analyses": date_findings,
                "integrated_report": integrated_report
            }
            
        return {
            "agent": self.name,
            "findings": integrated_reports
        }
