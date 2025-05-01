import os
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# 設定ファイルをインポート
from config import *

# 異常検知器をインポート
from detection.anomaly_detector import AnomalyDetector

# ユーティリティ関数をインポート
from utils.data_utils import load_sample_data, save_sample_data

# LLMクライアントをインポート
from utils.llm_clients import OpenAIClient, HuggingFaceClient, MockLLMClient

# エージェントをインポート
from agents.web_agent import WebInformationAgent
from agents.knowledge_agent import KnowledgeBaseAgent
from agents.crosscheck_agent import CrossCheckAgent
from agents.report_agent import ReportIntegrationAgent
from agents.manager_agent import ManagerAgent

# データを準備
def prepare_data(use_sample, file_path=None):
    """データを準備"""
    if use_sample:
        # サンプルデータを使用
        df = load_sample_data(START_DATE, END_DATE)
    else:
        # アップロードされたファイルを使用
        if file_path is None or file_path == "":
            raise ValueError("ファイルがアップロードされていません")
        
        # ファイルの拡張子を確認
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("対応していないファイル形式です。CSVまたはExcelファイルをアップロードしてください。")
        
        # 日付カラムを確認
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # 最初のカラムが日付と仮定
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        
        # 値のカラムを確認
        if 'Close' not in df.columns:
            # 2番目のカラムが値と仮定
            df.rename(columns={df.columns[1]: 'Close'}, inplace=True)
    
    return df

# 異常検知を実行
def detect_anomalies(df, method, threshold):
    """異常検知を実行"""
    detector = AnomalyDetector(method=method, threshold=float(threshold))
    return detector.detect(df)

# プロットを作成
def create_plot(df, anomalies):
    """時系列データと検出された異常値のプロットを作成"""
    fig = go.Figure()
    
    # 元のデータをプロット
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='時系列データ'
    ))
    
    # 異常値をプロット
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['Date'],
            y=anomalies['Close'],
            mode='markers',
            marker=dict(color='red', size=10),
            name='検出された異常'
        ))
    
    fig.update_layout(
        title='時系列データと検出された異常',
        xaxis_title='日付',
        yaxis_title='値',
        template='plotly_white',
        height=500
    )
    
    return fig

# LLMクライアントを取得
def get_llm_client(provider):
    """指定されたプロバイダに基づいてLLMクライアントを取得"""
    if provider == "openai":
        return OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    elif provider == "huggingface":
        return HuggingFaceClient(api_key=HF_API_KEY, model=HF_MODEL)
    else:
        return MockLLMClient()

# エージェントシステムを実行
def run_agent_system(anomalies, llm_provider, enabled_agents):
    """エージェントシステムを実行して結果を返す"""
    # LLMクライアントを取得
    llm_client = get_llm_client(llm_provider)
    
    # 重要な異常のみに絞る（パフォーマンス向上のため）
    # 最大3件の異常に制限
    if len(anomalies) > 3:
        # pct_changeの絶対値が大きい順にソート
        anomalies = anomalies.iloc[anomalies['pct_change'].abs().argsort()[::-1][:3]].copy()
    
    # エージェントを初期化
    agents = []
    
    if 'web' in enabled_agents:
        web_agent = WebInformationAgent(llm_client=llm_client)
        agents.append(web_agent)
    
    if 'knowledge' in enabled_agents:
        knowledge_agent = KnowledgeBaseAgent(llm_client=llm_client)
        agents.append(knowledge_agent)
    
    if 'crosscheck' in enabled_agents:
        crosscheck_agent = CrossCheckAgent(llm_client=llm_client)
        agents.append(crosscheck_agent)
    
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
    if 'report' in enabled_agents:
        report_agent = ReportIntegrationAgent(llm_client=llm_client)
        context['agent_findings'] = agent_findings
        print("レポート統合エージェントを実行中...")
        report_agent_result = report_agent.process(anomalies, context)
        agent_findings[report_agent.name] = report_agent_result
    
    # 管理者エージェントを実行
    if 'manager' in enabled_agents:
        manager_agent = ManagerAgent(llm_client=llm_client)
        print("管理者エージェントを実行中...")
        manager_agent_result = manager_agent.process(anomalies, context)
        agent_findings[manager_agent.name] = manager_agent_result
    
    return agent_findings

# 分析を実行
def run_analysis(data_source, file_path, detection_method, threshold, llm_provider, 
                use_web_agent, use_knowledge_agent, use_crosscheck_agent, 
                use_report_agent, use_manager_agent):
    try:
        # データを準備
        use_sample = (data_source == "sample")
        df = prepare_data(use_sample, file_path)
        
        # 異常検知を実行
        anomalies = detect_anomalies(df, detection_method, threshold)
        
        # プロットを作成
        plot = create_plot(df, anomalies)
        
        # 異常データを表として整形
        if anomalies.empty:
            anomalies_df = pd.DataFrame(columns=['日付', '値', '変化率 (%)'])
        else:
            anomalies_df = anomalies.copy()
            anomalies_df = anomalies_df[['Date', 'Close', 'pct_change']]
            anomalies_df.columns = ['日付', '値', '変化率 (%)']
            anomalies_df['日付'] = anomalies_df['日付'].dt.strftime('%Y-%m-%d')
            anomalies_df['変化率 (%)'] = anomalies_df['変化率 (%)'].round(2)
        
        # 有効なエージェントのリストを作成
        enabled_agents = []
        if use_web_agent:
            enabled_agents.append('web')
        if use_knowledge_agent:
            enabled_agents.append('knowledge')
        if use_crosscheck_agent:
            enabled_agents.append('crosscheck')
        if use_report_agent:
            enabled_agents.append('report')
        if use_manager_agent:
            enabled_agents.append('manager')
        
        # エージェント分析を実行
        agent_findings = {}
        if enabled_agents and not anomalies.empty:
            agent_findings = run_agent_system(anomalies, llm_provider, enabled_agents)
        
        # 分析結果を返す
        return plot, f"検出された異常: {len(anomalies)}件", anomalies_df, agent_findings
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return None, f"エラーが発生しました: {str(e)}", pd.DataFrame(), {}

# エージェント結果をHTMLとして整形
def format_agent_findings(agent_findings):
    if not agent_findings:
        return "エージェント分析結果はありません。"
    
    html = ""
    
    # 最終評価が存在する場合はそれを優先表示
    if "管理者エージェント" in agent_findings:
        manager_findings = agent_findings["管理者エージェント"].get("findings", {})
        if manager_findings:
            html += "<h3>最終評価</h3>"
            for date, data in manager_findings.items():
                html += f"<div style='margin-bottom: 15px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
                html += f"<h4>日付: {date} (変化率: {data.get('anomaly_details', {}).get('pct_change', 'N/A')}%)</h4>"
                html += f"<pre style='white-space: pre-wrap;'>{data.get('final_assessment', 'N/A')}</pre>"
                html += f"</div>"
    
    # 統合レポートが存在する場合はそれを表示
    if "レポート統合エージェント" in agent_findings:
        report_findings = agent_findings["レポート統合エージェント"].get("findings", {})
        if report_findings:
            html += "<h3>統合レポート</h3>"
            for date, data in report_findings.items():
                html += f"<div style='margin-bottom: 15px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
                html += f"<h4>日付: {date} (変化率: {data.get('anomaly_details', {}).get('pct_change', 'N/A')}%)</h4>"
                html += f"<pre style='white-space: pre-wrap;'>{data.get('integrated_report', 'N/A')}</pre>"
                html += f"</div>"
    
    # 個別のエージェント結果も表示
    html += "<h3>個別エージェント分析</h3>"
    
    for agent_name, findings in agent_findings.items():
        if agent_name not in ["レポート統合エージェント", "管理者エージェント"]:
            html += f"<details style='margin-bottom: 10px;'>"
            html += f"<summary style='cursor: pointer; padding: 5px; background-color: #f8f9fa;'>{agent_name}</summary>"
            html += f"<div style='padding: 10px; border: 1px solid #ddd; border-top: none;'>"
            
            if "error" in findings:
                html += f"<p>エラー: {findings['error']}</p>"
            elif "findings" in findings:
                for date, data in findings["findings"].items():
                    if "llm_analysis" in data:
                        html += f"<h4>日付: {date}</h4>"
                        html += f"<pre style='white-space: pre-wrap;'>{data['llm_analysis']}</pre>"
                        html += "<hr>"
            
            html += "</div>"
            html += "</details>"
    
    return html

# Gradio UIの作成
def create_gradio_ui():
    with gr.Blocks(title="異常検知分析システム") as app:
        gr.Markdown("# マルチエージェントLLM異常検知分析システム")
        gr.Markdown("時系列データの異常を検出し、LLMエージェントが分析します。")
        
        # 分析設定
        with gr.Group():
            gr.Markdown("## 分析設定")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### データ設定")
                    data_source = gr.Radio(
                        label="データソース",
                        choices=["sample", "upload"],
                        value="sample",
                        info="サンプルデータを使用するか、ファイルをアップロードするか選択してください"
                    )
                    file_path = gr.File(
                        label="時系列データファイル (CSVまたはExcel)",
                        type="filepath",
                        visible=False
                    )
                    
                    # データソースが変更されたときのイベントハンドラ
                    def update_file_visibility(choice):
                        return gr.update(visible=(choice == "upload"))
                    
                    data_source.change(fn=update_file_visibility, inputs=data_source, outputs=file_path)
                    
                with gr.Column():
                    gr.Markdown("### 異常検知設定")
                    detection_method = gr.Dropdown(
                        label="検出方法",
                        choices=["z_score", "iqr", "moving_avg"],
                        value="z_score",
                        info="Z-scoreは標準偏差ベース、IQRは四分位数ベース、移動平均はトレンドからの乖離を検出します"
                    )
                    threshold = gr.Slider(
                        label="検出閾値",
                        minimum=1.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        info="値が大きいほど検出される異常が少なくなります"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### LLM設定")
                    llm_provider = gr.Radio(
                        label="LLMプロバイダ",
                        choices=["openai", "huggingface", "mock"],
                        value="mock",
                        info="OpenAIまたはHugging Faceを使用する場合はAPIキーが必要です。Mockはデモ用のモックレスポンスを返します。"
                    )
                
                with gr.Column():
                    gr.Markdown("### エージェント設定")
                    use_web_agent = gr.Checkbox(label="Web情報エージェント", value=True)
                    use_knowledge_agent = gr.Checkbox(label="知識ベースエージェント", value=True)
                    use_crosscheck_agent = gr.Checkbox(label="クロスチェックエージェント", value=True)
                    use_report_agent = gr.Checkbox(label="レポート統合エージェント", value=True)
                    use_manager_agent = gr.Checkbox(label="管理者エージェント", value=True)
            
            analyze_btn = gr.Button("異常検知を実行", variant="primary")
            status_text = gr.Markdown("")
            
        # 分析結果
        with gr.Group() as results_container:
            gr.Markdown("## 分析結果")
            
            # タブを使用して結果を表示
            with gr.Tabs() as result_tabs:
                # プロットタブ
                with gr.TabItem("プロット"):
                    plot_output = gr.Plot()
                
                # 異常テーブルタブ
                with gr.TabItem("検出された異常"):
                    anomaly_count = gr.Markdown("")
                    anomaly_table = gr.DataFrame()
                
                # エージェント分析タブ
                with gr.TabItem("エージェント分析"):
                    agent_results = gr.HTML("")
        
        # 実行前の状態表示
        def set_running_status():
            return "分析を実行中です。結果が表示されるまでお待ちください..."
        
        # 分析実行後の結果表示
        def show_results(plot, count, table, findings):
            agent_html = format_agent_findings(findings)
            return (
                "",  # ステータステキストをクリア
                plot,  # プロット
                count,  # 異常カウント
                table,  # 異常テーブル
                agent_html  # エージェント分析結果
            )
        
        # 実行ボタンイベント
        analyze_btn.click(
            fn=set_running_status,
            inputs=None,
            outputs=status_text
        ).then(
            fn=run_analysis,
            inputs=[
                data_source, file_path,
                detection_method, threshold,
                llm_provider,
                use_web_agent, use_knowledge_agent, use_crosscheck_agent, use_report_agent, use_manager_agent
            ],
            outputs=[plot_output, anomaly_count, anomaly_table, agent_results]
        ).then(
            fn=show_results,
            inputs=[plot_output, anomaly_count, anomaly_table, agent_results],
            outputs=[status_text, plot_output, anomaly_count, anomaly_table, agent_results]
        )
    
    return app

# メイン処理
if __name__ == "__main__":
    app = create_gradio_ui()
    app.queue().launch(share=False)
