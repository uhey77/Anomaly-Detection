import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import matplotlib.pyplot as plt
import json

# .envファイルから環境変数をロード
load_dotenv()

# 設定ファイルをインポート
from config import *

# 異常検知器をインポート
from detection.anomaly_detector import AnomalyDetector

# ユーティリティ関数をインポート
from utils.data_utils import load_sample_data, save_sample_data, plot_anomalies

# LLMクライアントをインポート
from utils.llm_clients import OpenAIClient, HuggingFaceClient, MockLLMClient

# エージェントをインポート
from agents.web_agent import WebInformationAgent
from agents.knowledge_agent import KnowledgeBaseAgent
from agents.crosscheck_agent import CrossCheckAgent
from agents.report_agent import ReportIntegrationAgent
from agents.manager_agent import ManagerAgent

# メイン処理
def main():
    print("マルチエージェントLLM異常検知フレームワーク開始")
    
    # LLMクライアントを初期化
    llm_client = get_llm_client()
    
    # サンプルデータを準備
    df = prepare_sample_data()
    
    # 異常検知器を初期化して実行
    detector = AnomalyDetector(method=ANOMALY_METHOD, threshold=ANOMALY_THRESHOLD)
    anomalies = detector.detect(df)
    
    print(f"検出された異常: {len(anomalies)}")
    
    # 異常をプロット
    plot_anomalies(df, anomalies, output_file=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # エージェントシステムを実行
    run_agent_system(anomalies, llm_client)
    
    print("処理完了")

def get_llm_client():
    """環境設定に基づいてLLMクライアントを取得"""
    if LLM_PROVIDER == "openai":
        return OpenAIClient()
    elif LLM_PROVIDER == "huggingface":
        return HuggingFaceClient()
    else:
        return MockLLMClient()

def prepare_sample_data():
    """サンプルデータを準備"""
    # データディレクトリを確認
    os.makedirs("data", exist_ok=True)
    
    # サンプルデータファイルのパス
    sample_file_path = "data/sp500_sample.csv"
    
    # ファイルが存在する場合は読み込み、存在しない場合は作成
    if os.path.exists(sample_file_path):
        print(f"サンプルデータを {sample_file_path} から読み込みます")
        df = pd.read_csv(sample_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print("サンプルデータを作成します")
        df = load_sample_data(START_DATE, END_DATE)
        save_sample_data(df)
    
    return df

def run_agent_system(anomalies, llm_client):
    """エージェントシステムを実行"""
    # エージェントを初期化
    agents = []
    
    if ENABLE_WEB_AGENT:
        web_agent = WebInformationAgent(llm_client=llm_client)
        agents.append(web_agent)
    
    if ENABLE_KNOWLEDGE_AGENT:
        knowledge_agent = KnowledgeBaseAgent(llm_client=llm_client)
        agents.append(knowledge_agent)
    
    if ENABLE_CROSSCHECK_AGENT:
        crosscheck_agent = CrossCheckAgent(llm_client=llm_client)
        agents.append(crosscheck_agent)
    
    if ENABLE_REPORT_AGENT:
        report_agent = ReportIntegrationAgent(llm_client=llm_client)
        agents.append(report_agent)
    
    if ENABLE_MANAGER_AGENT:
        manager_agent = ManagerAgent(llm_client=llm_client)
        agents.append(manager_agent)
    
    # コンテキスト準備
    context = {
        'reference_data': {
            # 参照用の追加データがあればここに追加
        }
    }
    
    # エージェント結果を保存
    agent_findings = {}
    
    # 各エージェントを実行
    for agent in agents:
        if agent.name in ["レポート統合エージェント", "管理者エージェント"]:
            # これらのエージェントは他のエージェントの結果を必要とするので、後で実行
            continue
        
        print(f"{agent.name}を実行中...")
        agent_result = agent.process(anomalies, context)
        agent_findings[agent.name] = agent_result
    
    # 統合エージェントを実行
    if ENABLE_REPORT_AGENT:
        context['agent_findings'] = agent_findings
        print("レポート統合エージェントを実行中...")
        report_agent_result = report_agent.process(anomalies, context)
        agent_findings[report_agent.name] = report_agent_result
    
    # 管理者エージェントを実行
    if ENABLE_MANAGER_AGENT:
        print("管理者エージェントを実行中...")
        manager_agent_result = manager_agent.process(anomalies, context)
        agent_findings[manager_agent.name] = manager_agent_result
    
    # 結果を保存
    save_results(anomalies, agent_findings)

def save_results(anomalies, agent_findings):
    """分析結果を保存"""
    # 結果ディレクトリを作成
    os.makedirs("results", exist_ok=True)
    
    # 現在のタイムスタンプを取得
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 異常データを保存
    anomalies_csv = anomalies.copy()
    anomalies_csv['Date'] = anomalies_csv['Date'].dt.strftime('%Y-%m-%d')
    anomalies_csv.to_csv(f"results/anomalies_{timestamp}.csv", index=False)
    
    # 結果をJSONとして保存
    results = {
        "timestamp": timestamp,
        "anomalies_count": len(anomalies),
        "anomalies_dates": anomalies['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "agent_findings": {}
    }
    
    # JSONシリアライズ可能なデータ構造に変換
    for agent_name, findings in agent_findings.items():
        agent_result = {}
        agent_result["agent"] = findings.get("agent", "")
        
        if "error" in findings:
            agent_result["error"] = findings["error"]
        elif "findings" in findings:
            agent_findings_json = {}
            for date, data in findings["findings"].items():
                # DataFrameや非シリアライズ可能なオブジェクトを処理
                date_findings = {}
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool, type(None))):
                        date_findings[key] = value
                    elif isinstance(value, dict):
                        # 辞書の場合は再帰的にシリアライズ可能なオブジェクトに変換
                        date_findings[key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (str, int, float, bool, type(None))):
                                date_findings[key][sub_key] = sub_value
                            else:
                                date_findings[key][sub_key] = str(sub_value)
                    else:
                        date_findings[key] = str(value)
                agent_findings_json[date] = date_findings
            agent_result["findings"] = agent_findings_json
        
        results["agent_findings"][agent_name] = agent_result
    
    with open(f"results/analysis_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"結果をresults/analysis_{timestamp}.jsonに保存しました")
    
    # テキストレポートも生成
    with open(f"results/report_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(f"# 異常検知分析レポート - {timestamp}\n\n")
        f.write(f"検出された異常: {len(anomalies)}\n")
        f.write("日付:\n")
        for date in anomalies['Date'].dt.strftime('%Y-%m-%d'):
            f.write(f"- {date}\n")
        f.write("\n")
        
        # 管理者エージェントの最終評価があれば追加
        if "管理者エージェント" in agent_findings and "findings" in agent_findings["管理者エージェント"]:
            f.write("## 最終評価\n\n")
            for date, data in agent_findings["管理者エージェント"]["findings"].items():
                f.write(f"### {date}\n")
                f.write(f"変化率: {data.get('anomaly_details', {}).get('pct_change', 'N/A')}%\n\n")
                f.write(f"{data.get('final_assessment', 'N/A')}\n\n")
    
    print(f"レポートをresults/report_{timestamp}.txtに保存しました")

if __name__ == "__main__":
    main()
