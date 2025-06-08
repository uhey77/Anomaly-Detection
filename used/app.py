import os
import gradio as gr
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # バックエンドを無表示に設定

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

# LLMクライアントを取得
def get_llm_client(provider):
    """指定されたプロバイダに基づいてLLMクライアントを取得"""
    if provider == "openai":
        return OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    elif provider == "huggingface":
        return HuggingFaceClient(api_key=HF_API_KEY, model=HF_MODEL)
    else:
        return MockLLMClient()

# データを準備
def prepare_data(use_sample, file_path=None, start_date=None, end_date=None):
    """データを準備"""
    if use_sample:
        # サンプルデータを使用
        df = load_sample_data(start_date or START_DATE, end_date or END_DATE)
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

# データと異常値のプロットを作成
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

# エージェントシステムを実行
def run_agent_system(anomalies, llm_provider, enabled_agents):
    """エージェントシステムを実行して結果を返す"""
    # LLMクライアントを取得
    llm_client = get_llm_client(llm_provider)
    
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
    
    if 'report' in enabled_agents:
        report_agent = ReportIntegrationAgent(llm_client=llm_client)
        agents.append(report_agent)
    
    if 'manager' in enabled_agents:
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
    if 'report' in enabled_agents:
        context['agent_findings'] = agent_findings
        print("レポート統合エージェントを実行中...")
        report_agent_result = report_agent.process(anomalies, context)
        agent_findings[report_agent.name] = report_agent_result
    
    # 管理者エージェントを実行
    if 'manager' in enabled_agents:
        print("管理者エージェントを実行中...")
        manager_agent_result = manager_agent.process(anomalies, context)
        agent_findings[manager_agent.name] = manager_agent_result
    
    return agent_findings

# 異常検知Webアプリのメイン関数
def anomaly_detection_app(
    data_source, file_path, 
    detection_method, threshold,
    llm_provider, 
    use_web_agent, use_knowledge_agent, use_crosscheck_agent, use_report_agent, use_manager_agent,
    progress=gr.Progress()  # プログレスコンポーネントを受け取る
):
    """
    異常検知Webアプリのメイン関数
    """
    try:
        # 進捗表示を更新
        progress(0, desc="準備中...")
        status_update = "<p><strong>データ準備中...</strong> (0%)</p>"
        
        # データを準備
        use_sample = (data_source == "sample")
        df = prepare_data(use_sample, file_path)
        
        progress(0.2, desc="異常検知実行中...")
        status_update = "<p><strong>異常検知実行中...</strong> (20%)</p>"
        
        # 異常検知を実行
        anomalies = detect_anomalies(df, detection_method, threshold)
        
        # プロットを作成
        progress(0.4, desc="プロット作成中...")
        status_update = "<p><strong>プロット作成中...</strong> (40%)</p>"
        plot = create_plot(df, anomalies)
        
        # 異常テーブルを作成
        if anomalies.empty:
            anomalies_html = "<p>異常は検出されませんでした。</p>"
            
            # 処理完了
            progress(1.0, desc="完了")
            status_update = f"<p><strong>分析完了</strong> - 異常は検出されませんでした。({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</p>"
            
            # 結果HTMLを作成（プロットだけのシンプルなバージョン）
            # anomaly_detection_app関数内の結果HTML生成部分を修正
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>異常検知分析結果</title>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                    .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
                    .tab button:hover {{ background-color: #ddd; }}
                    .tab button.active {{ background-color: #ccc; }}
                    .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
                    .active-tab {{ display: block; }}
                    pre {{ white-space: pre-wrap; }}
                    .success-alert {{ padding: 15px; background-color: #d4edda; color: #155724; margin-bottom: 15px; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <h2>異常検知分析結果</h2>
                <div class="success-alert">
                    <strong>処理完了!</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                
                <div class="tab">
                    <button class="tablinks active" onclick="openTab(event, 'Plot')">プロット</button>
                    <button class="tablinks" onclick="openTab(event, 'Anomalies')">検出された異常</button>
                    <button class="tablinks" onclick="openTab(event, 'Agents')">エージェント分析</button>
                </div>
                
                <div id="Plot" class="tabcontent active-tab">
                    <div id="plotly-div"></div>
                </div>
                
                <div id="Anomalies" class="tabcontent">
                    {anomalies_html}
                </div>
                
                <div id="Agents" class="tabcontent">
                    {agents_html}
                </div>

                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script>
                    // プロットを作成
                    var plotlyData = {plot.to_json()};
                    Plotly.newPlot('plotly-div', JSON.parse(plotlyData));
                    
                    // タブ切り替え関数
                    function openTab(evt, tabName) {{
                        var i, tabcontent, tablinks;
                        
                        // すべてのtabcontentを非表示
                        tabcontent = document.getElementsByClassName("tabcontent");
                        for (i = 0; i < tabcontent.length; i++) {{
                            tabcontent[i].style.display = "none";
                            tabcontent[i].classList.remove("active-tab");
                        }}
                        
                        // すべてのtablinksからactiveクラスを削除
                        tablinks = document.getElementsByClassName("tablinks");
                        for (i = 0; i < tablinks.length; i++) {{
                            tablinks[i].className = tablinks[i].className.replace(" active", "");
                        }}
                        
                        // クリックされたタブとコンテンツをアクティブに
                        document.getElementById(tabName).style.display = "block";
                        document.getElementById(tabName).classList.add("active-tab");
                        evt.currentTarget.className += " active";
                    }}
                </script>
            </body>
            </html>
            """
            
            return status_update, html
        
        # 表示用にデータを整形
        progress(0.5, desc="表形式データ準備中...")
        status_update = "<p><strong>表形式データ準備中...</strong> (50%)</p>"
        anomalies_display = anomalies.copy()
        anomalies_display['Date'] = anomalies_display['Date'].dt.strftime('%Y-%m-%d')
        anomalies_display['pct_change'] = anomalies_display['pct_change'].round(2)
        
        # HTML表を作成
        anomalies_html = f"<h3>検出された異常 ({len(anomalies)}件)</h3>"
        anomalies_html += anomalies_display[['Date', 'Close', 'pct_change']].to_html(
            index=False, 
            float_format='%.2f',
            classes='table table-striped',
            columns=['Date', 'Close', 'pct_change'],
            header=['日付', '値', '変化率 (%)']
        )
        
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
        if not enabled_agents:
            agents_html = "<p>エージェント分析は無効になっています。</p>"
            
            # 処理完了
            progress(1.0, desc="完了")
            status_update = f"<p><strong>分析完了！</strong> ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</p>"
        else:
            # エージェントシステムを実行
            progress(0.6, desc="エージェント分析実行中...")
            status_update = "<p><strong>エージェント分析実行中...</strong> (60%)</p>"
            agent_findings = run_agent_system(anomalies, llm_provider, enabled_agents)
            
            # 結果をHTMLに変換
            progress(0.8, desc="結果レポート生成中...")
            status_update = "<p><strong>結果レポート生成中...</strong> (80%)</p>"
            agents_html = "<h3>エージェント分析結果</h3>"
            
            # 最終評価が存在する場合はそれを優先表示
            if 'manager' in enabled_agents and "管理者エージェント" in agent_findings:
                manager_findings = agent_findings["管理者エージェント"].get("findings", {})
                if manager_findings:
                    agents_html += "<h4>最終評価</h4>"
                    for date, data in manager_findings.items():
                        agents_html += f"<div class='card mb-3'>"
                        agents_html += f"<div class='card-header'><strong>日付: {date}</strong> (変化率: {data.get('anomaly_details', {}).get('pct_change', 'N/A')}%)</div>"
                        agents_html += f"<div class='card-body'><pre>{data.get('final_assessment', 'N/A')}</pre></div>"
                        agents_html += f"</div>"
            
            # 統合レポートが存在する場合はそれを表示
            if 'report' in enabled_agents and "レポート統合エージェント" in agent_findings:
                report_findings = agent_findings["レポート統合エージェント"].get("findings", {})
                if report_findings:
                    agents_html += "<h4>統合レポート</h4>"
                    for date, data in report_findings.items():
                        agents_html += f"<div class='card mb-3'>"
                        agents_html += f"<div class='card-header'><strong>日付: {date}</strong> (変化率: {data.get('anomaly_details', {}).get('pct_change', 'N/A')}%)</div>"
                        agents_html += f"<div class='card-body'><pre>{data.get('integrated_report', 'N/A')}</pre></div>"
                        agents_html += f"</div>"
            
            # 個別のエージェント結果も表示
            agents_html += "<h4>個別エージェント分析</h4>"
            agents_html += "<div class='accordion' id='agentsAccordion'>"
            
            accordion_items = []
            for i, (agent_name, findings) in enumerate(agent_findings.items()):
                if agent_name not in ["レポート統合エージェント", "管理者エージェント"]:
                    accordion_items.append((agent_name, findings))
            
            for i, (agent_name, findings) in enumerate(accordion_items):
                agents_html += f"""
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading{i}">
                        <button class="accordion-button {'collapsed' if i > 0 else ''}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{i}" aria-expanded="{str(i == 0).lower()}" aria-controls="collapse{i}">
                            {agent_name}
                        </button>
                    </h2>
                    <div id="collapse{i}" class="accordion-collapse collapse {'show' if i == 0 else ''}" aria-labelledby="heading{i}" data-bs-parent="#agentsAccordion">
                        <div class="accordion-body">
                """
                
                if "error" in findings:
                    agents_html += f"<p>エラー: {findings['error']}</p>"
                elif "findings" in findings:
                    for date, data in findings["findings"].items():
                        if "llm_analysis" in data:
                            agents_html += f"<h5>日付: {date}</h5>"
                            agents_html += f"<pre>{data['llm_analysis']}</pre>"
                            agents_html += "<hr>"
                
                agents_html += """
                        </div>
                    </div>
                </div>
                """
            
            agents_html += "</div>"  # アコーディオン終了
            
            # 処理完了
            progress(1.0, desc="完了")
            status_update = f"<p><strong>分析完了！</strong> ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})</p>"
        
        # 最終的なHTMLを組み立て
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>異常検知分析結果</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .container {{ max-width: 1200px; margin-top: 20px; }}
                pre {{ white-space: pre-wrap; }}
                .tab-content {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>異常検知分析結果</h2>
                <div class="alert alert-success">
                    <strong>処理完了!</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                
                <ul class="nav nav-tabs" id="resultTabs">
                    <li class="nav-item">
                        <a class="nav-link active" data-bs-toggle="tab" href="#plot">プロット</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" data-bs-toggle="tab" href="#anomalies">検出された異常</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" data-bs-toggle="tab" href="#agents">エージェント分析</a>
                    </li>
                </ul>
                
                <div class="tab-content">
                    <div id="plot" class="tab-pane fade show active">
                        <div id="plotly-div" class="plotly-plot"></div>
                    </div>
                    <div id="anomalies" class="tab-pane fade">
                        {anomalies_html}
                    </div>
                    <div id="agents" class="tab-pane fade">
                        {agents_html}
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script>
                // Plotlyプロットを埋め込み
                var plotlyData = {plot.to_json()};
                Plotly.newPlot('plotly-div', JSON.parse(plotlyData));
                
                // タブ機能が正しく動作するようにするJavaScript
                document.addEventListener('DOMContentLoaded', function() {{
                    var triggerTabList = [].slice.call(document.querySelectorAll('#resultTabs a'));
                    triggerTabList.forEach(function(triggerEl) {{
                        triggerEl.addEventListener('click', function(event) {{
                            event.preventDefault();
                            
                            // すべてのタブとコンテンツを非アクティブに
                            document.querySelectorAll('.nav-link').forEach(function(tab) {{
                                tab.classList.remove('active');
                            }});
                            document.querySelectorAll('.tab-pane').forEach(function(pane) {{
                                pane.classList.remove('show', 'active');
                            }});
                            
                            // クリックされたタブとそのコンテンツをアクティブに
                            this.classList.add('active');
                            var target = document.querySelector(this.getAttribute('href'));
                            if (target) {{
                                target.classList.add('show', 'active');
                            }}
                        }});
                    }});
                }});
            </script>
        </body>
        </html>
        """
        
        return status_update, html
    
    except Exception as e:
        # エラー表示
        progress(1.0, desc="エラー発生")
        status_update = f"<p><strong>エラーが発生しました</strong> - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        error_html = f"""
        <div class="alert alert-danger">
            <h4>エラーが発生しました</h4>
            <p>{str(e)}</p>
        </div>
        """
        return status_update, error_html

# Gradio UIの作成
def create_gradio_ui():
    with gr.Blocks(title="異常検知分析システム", theme=gr.themes.Base(), css="""
        #result_container {
            width: 100%;
            min-height: 600px;
        }
        .full-width {
            width: 100% !important;
        }
        iframe {
            width: 100%;
            height: 800px;
            border: none;
        }
    """) as app:
        gr.Markdown("# マルチエージェントLLM異常検知分析システム")
        gr.Markdown("時系列データの異常を検出し、LLMエージェントが分析します。")
        
        with gr.Tab("分析設定"):
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
                        type="filepath",  # "file"から"filepath"に変更
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
            
        with gr.Tab("分析結果"):
            with gr.Row():
                status_html = gr.HTML("<p>実行開始前です。「異常検知を実行」ボタンをクリックしてください。</p>")

            with gr.Row():
                progress = gr.Progress()

            result_html = gr.HTML("", elem_id="result_container", elem_classes=["full-width"])

        # 実行ボタンのイベントハンドラ
        analyze_btn.click(
            fn=anomaly_detection_app,
            inputs=[
                data_source, file_path,
                detection_method, threshold,
                llm_provider,
                use_web_agent, use_knowledge_agent, use_crosscheck_agent, use_report_agent, use_manager_agent
            ],
            outputs=[status_html, result_html]
        )

    return app

# メイン処理
if __name__ == "__main__":
    app = create_gradio_ui()
    app.launch(share=False)
