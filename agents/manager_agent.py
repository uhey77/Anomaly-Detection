from .base_agent import BaseAgent

class ManagerAgent(BaseAgent):
    """統合レポートをレビューし、最終推奨事項を行う管理者エージェント"""
    
    def __init__(self, name="管理者エージェント", llm_client=None):
        """
        管理者エージェントを初期化
        
        Args:
            name (str): エージェント名
            llm_client: LLMクライアントインスタンス
        """
        super().__init__(name, llm_client)
        
    def process(self, anomaly_data, context=None):
        """
        統合レポートをレビューし、最終推奨事項を行う
        
        Args:
            anomaly_data (pd.DataFrame): 異常を含むデータ
            context (dict): 統合レポートを含む追加コンテキスト
            
        Returns:
            dict: 最終評価と推奨事項
        """
        if not context or 'agent_findings' not in context or 'レポート統合エージェント' not in context['agent_findings']:
            return {
                "agent": self.name,
                "error": "コンテキストに統合レポートが提供されていません"
            }
            
        report_findings = context['agent_findings']['レポート統合エージェント']['findings']
        
        # 各異常の最終評価を準備
        final_assessments = {}
        
        for anomaly_date, report_data in report_findings.items():
            integrated_report = report_data.get('integrated_report', '統合レポートなし')
            anomaly_details = report_data.get('anomaly_details', {})
            
            # 最終評価のためのLLMクエリ
            llm_prompt = f"""
            上級金融市場専門家として、市場異常に関するこの統合レポートをレビューし、最終評価を提供してください:
            
            異常詳細:
            日付: {anomaly_details.get('date')}
            S&P 500値: {anomaly_details.get('value')}
            変化率: {anomaly_details.get('pct_change')}%
            
            統合レポート:
            {integrated_report}
            
            以下を提供してください:
            1. エグゼクティブサマリー: 異常とその原因の2〜3文の要約
            2. 妥当性評価: これは真の異常ですか、それとも潜在的に誤ったデータですか？（高/中/低信頼度）
            3. 重要性評価: このイベントは広範な市場のコンテキストでどの程度重要ですか？（重大/高/中/低）
            4. 行動推奨: この異常に対してどのような行動を取るべきですか（もしあれば）？
            5. 学習ポイント: 将来の市場監視のために、この異常から何を学ぶことができますか？
            
            あなたの評価は権威あり、簡潔で、実用的であるべきです。
            """
            
            final_assessment = self.query_llm(llm_prompt)
            
            final_assessments[anomaly_date] = {
                "anomaly_details": anomaly_details,
                "integrated_report": integrated_report,
                "final_assessment": final_assessment
            }
            
        return {
            "agent": self.name,
            "findings": final_assessments
        }
