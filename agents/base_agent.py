from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """システム内のすべてのエージェントの基本クラス"""
    
    def __init__(self, name, llm_client=None):
        """
        基本エージェントを初期化
        
        Args:
            name (str): エージェント名
            llm_client: LLMクライアントインスタンス
        """
        self.name = name
        self.llm_client = llm_client
        
    @abstractmethod
    def process(self, data, context=None):
        """
        データを処理し、結果を返す
        
        Args:
            data: 入力データ
            context: 追加コンテキスト
            
        Returns:
            dict: 結果
        """
        pass
        
    def query_llm(self, prompt):
        """
        指定のプロンプトでLLMにクエリを実行
        
        Args:
            prompt (str): 入力プロンプト
            
        Returns:
            str: LLMレスポンス
        """
        if self.llm_client is None:
            return "LLMクライアントが設定されていません"
        
        try:
            # 特定のLLMクライアントに基づいて実装される
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            return f"LLMクエリエラー: {str(e)}"
