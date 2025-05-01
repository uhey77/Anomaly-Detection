import requests
from bs4 import BeautifulSoup
from .base_agent import BaseAgent

class WebInformationAgent(BaseAgent):
    """異常に関連する情報をWebから検索するエージェント"""
    
    def __init__(self, name="Web情報エージェント", llm_client=None, api_key=None):
        """
        Web情報エージェントを初期化
        
        Args:
            name (str): エージェント名
            llm_client: LLMクライアントインスタンス
            api_key (str, optional): Web検索用APIキー
        """
        super().__init__(name, llm_client)
        self.api_key = api_key
        
    def process(self, anomaly_data, context=None):
        """
        異常に関する情報をWebから検索
        
        Args:
            anomaly_data (pd.DataFrame): 異常を含むデータ
            context (dict, optional): 追加コンテキスト
            
        Returns:
            dict: Web情報の結果
        """
        findings = {}
        
        for idx, anomaly in anomaly_data.iterrows():
            date = anomaly['Date']
            value = anomaly['Close']
            pct_change = anomaly['pct_change']
            
            # 検索用の日付フォーマット
            search_date = date.strftime("%Y-%m-%d")
            
            # 検索クエリの作成
            search_query = f"株式市場 S&P 500 重要イベント {search_date} {pct_change:.2f}% 変動"
            
            # Web検索（簡易版 - 実際の実装ではプロパーAPIを使用）
            web_results = self._mock_web_search(search_query)
            
            # Web結果を分析するためのLLMクエリ
            llm_prompt = f"""
            株式市場の異常に関する次のWeb検索結果を分析してください:
            
            日付: {search_date}
            S&P 500値: {value}
            変化率: {pct_change:.2f}%
            
            Web検索結果:
            {web_results}
            
            この情報に基づいて、この市場異常の原因は何だと考えられますか？
            潜在的な原因と関連するコンテキストを含む簡潔な分析を提供してください。
            """
            
            llm_analysis = self.query_llm(llm_prompt)
            
            findings[search_date] = {
                "raw_search_results": web_results,
                "llm_analysis": llm_analysis
            }
            
        return {
            "agent": self.name,
            "findings": findings
        }
    
    def _mock_web_search(self, query):
        """
        モックWeb検索関数（実際のAPI呼び出しに置き換える）
        
        Args:
            query (str): 検索クエリ
            
        Returns:
            str: モック検索結果
        """
        # デモ用に、クエリに基づいてモック結果を返す
        if "1987-10-19" in query:
            return """
            1987-10-19の株式市場暴落に関する結果:
            1. ブラックマンデー: 1987年の株式市場暴落は、1987年10月下旬の数日間で起きた米国株価の急激で厳しい下落でした。
            2. 1987年10月19日、ダウ工業株30種平均は1日で22.6%下落しました。
            3. 暴落の原因は、プログラム取引、過大評価、流動性不足、投資家のパニックなど複数の要因が組み合わさったものでした。
            """
        elif "2008-10-13" in query:
            return """
            2008-10-13の株式市場イベントに関する結果:
            1. 2008年10月13日は、金融危機の中でS&P 500が1日で最大のパーセンテージ上昇を記録した日の一つでした。
            2. 世界中の政府が銀行システム支援計画を発表した後、市場は急騰しました。
            3. ダウ工業株30種平均は936ポイント（11.1%）上昇し、S&P 500は11.8%、ナスダックは11.8%上昇しました。
            """
        elif "2020-03-16" in query:
            return """
            2020-03-16の株式市場暴落に関する結果:
            1. 2020年3月16日は、COVID-19パンデミックによる株式市場の最悪の暴落の一つでした。
            2. S&P 500は約12%下落し、1987年のブラックマンデー以来で最悪の日となりました。
            3. この暴落は、連邦準備制度理事会の緊急利下げ（ほぼゼロ）と7000億ドルの量的緩和プログラム発表に続いて起こりました。
            """
        else:
            # 他の日付に対する一般的な応答
            return f"クエリに対する具体的な主要イベントは見つかりませんでした: {query}"
