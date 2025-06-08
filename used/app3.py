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
from utils.data_utils import load_sample_data, save_sample_data, load_multi_indicator_data

# LLMクライアントをインポート
from utils.llm_clients import OpenAIClient, HuggingFaceClient, MockLLMClient

# エージェントをインポート
from agents.web_agent import WebInformationAgent
from agents.knowledge_agent import KnowledgeBaseAgent
from agents.crosscheck_agent import CrossCheckAgent
from agents.report_agent import ReportIntegrationAgent
from agents.manager_agent import ManagerAgent

# 評価クラスをインポート
from evaluation import AnomalyEvaluator

# シグナル生成器と予測モデルをインポート
from utils.signal_generator import SignalGenerator
from models.time_series_models import LSTMModel, TimesFMModel
from models.forecasting_pipeline import ForecastingPipeline

# データを準備
def prepare_data(use_sample, file_path=None, include_extra_indicators=True):
    """データを準備"""
    if use_sample:
        # サンプルデータを使用
        if include_extra_indicators:
            # 複数指標を含むデータを読み込む
            data_dict = load_multi_indicator_data(START_DATE, END_DATE)
            df = data_dict['sp500']
            
            # 追加指標をマージ
            if 'volume' in data_dict:
                df = pd.merge(df, data_dict['volume'][['Date', 'Volume']], on='Date', how='left')
            
            if 'vix' in data_dict:
                df = pd.merge(df, data_dict['vix'][['Date', 'VIX']], on='Date', how='left')
            
            if 'usdjpy' in data_dict:
                df = pd.merge(df, data_dict['usdjpy'][['Date', 'USDJPY']], on='Date', how='left')
        else:
            # 通常のサンプルデータを読み込む
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
def detect_anomalies(df, method, threshold, extra_indicators=None):
    """異常検知を実行"""
    # 追加のパラメータを取得
    method_params = ANOMALY_PARAMS.get(method, {})
    
    # 追加の特徴量を設定
    extra_features = None
    if extra_indicators:
        extra_features = []
        # 出来高データがあれば追加
        if 'Volume' in df.columns:
            # 出来高の特徴量を計算
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            extra_features.append('volume_ratio')
        
        # VIXデータがあれば追加
        if 'VIX' in df.columns:
            extra_features.append('VIX')
        
        # ドル円データがあれば追加
        if 'USDJPY' in df.columns:
            # ドル円の変化率を計算
            df['usdjpy_pct_change'] = df['USDJPY'].pct_change() * 100
            extra_features.append('usdjpy_pct_change')
    
    detector = AnomalyDetector(method=method, threshold=float(threshold), **method_params)
    return detector.detect(df, extra_features=extra_features)

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

# 予測プロットを作成
def create_forecast_plot(df, anomalies, forecast_df, adjusted_forecast_df):
    """時系列データ、検出された異常、予測のプロットを作成"""
    fig = go.Figure()
    
    # 元のデータをプロット
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='実際の価格'
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
    
    # 通常予測をプロット
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['predicted_price'],
            mode='lines',
            line=dict(dash='dash', color='blue'),
            name='基本予測'
        ))
    
    # 異常調整済み予測をプロット
    if adjusted_forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=adjusted_forecast_df['Date'],
            y=adjusted_forecast_df['predicted_price'],
            mode='lines',
            line=dict(color='green'),
            name='異常調整済み予測'
        ))
    
    fig.update_layout(
        title='時系列データと将来予測',
        xaxis_title='日付',
        yaxis_title='価格',
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
def run_agent_system(anomalies, llm_provider, enabled_agents, reference_data=None):
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
        'reference_data': reference_data if reference_data else {}
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
                use_report_agent, use_manager_agent, include_extra_indicators=True,
                generate_signals=True, forecast_days=30):
    try:
        # データを準備
        use_sample = (data_source == "sample")
        df = prepare_data(use_sample, file_path, include_extra_indicators)
        
        # 異常検知を実行
        anomalies = detect_anomalies(df, detection_method, threshold, 
                                    include_extra_indicators)
        
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
        
        # 参照データを準備（クロスチェック用）
        reference_data = {}
        if include_extra_indicators:
            # 出来高データがあれば追加
            if 'Volume' in df.columns:
                volume_df = df[['Date', 'Volume']].copy()
                reference_data['volume'] = volume_df
            
            # VIXデータがあれば追加
            if 'VIX' in df.columns:
                vix_df = df[['Date', 'VIX']].copy()
                reference_data['vix'] = vix_df
            
            # ドル円データがあれば追加
            if 'USDJPY' in df.columns:
                usdjpy_df = df[['Date', 'USDJPY']].copy()
                reference_data['usdjpy'] = usdjpy_df
        
        # エージェント分析を実行
        agent_findings = {}
        if enabled_agents and not anomalies.empty:
            agent_findings = run_agent_system(anomalies, llm_provider, enabled_agents, reference_data)
        
        # シグナル生成と予測（オプション）
        signals_df = None
        forecast_df = None
        adjusted_forecast_df = None
        forecast_plot = None
        
        if generate_signals and not anomalies.empty:
            # シグナル生成と予測パイプラインを実行
            pipeline = ForecastingPipeline(
                forecasting_model=FORECASTING_MODEL,
                signal_threshold=SIGNAL_THRESHOLD,
                window_size=SIGNAL_WINDOW,
                lookback=FORECASTING_PARAMS[FORECASTING_MODEL].get('lookback', 60)
            )
            
            signals_df, forecast_df, adjusted_forecast_df = pipeline.process(
                df, anomalies, 
                train_model=True, 
                days_ahead=forecast_days, 
                adjustment_weight=0.5
            )
            
            # 予測プロットを作成
            forecast_plot = create_forecast_plot(df, anomalies, forecast_df, adjusted_forecast_df)
        
        # シグナルを表として整形
        signals_table = pd.DataFrame()
        if signals_df is not None and not signals_df.empty:
            signals_table = signals_df[signals_df['signal'] != 0][['Date', 'Close', 'pct_change', 'signal', 'anomaly_score']]
            if not signals_table.empty:
                signals_table.columns = ['日付', '価格', '変化率 (%)', 'シグナル(-1=売/1=買)', '異常強度']
                signals_table['日付'] = signals_table['日付'].dt.strftime('%Y-%m-%d')
                signals_table['変化率 (%)'] = signals_table['変化率 (%)'].round(2)
                signals_table['異常強度'] = signals_table['異常強度'].round(2)
        
        # 分析結果を返す
        return plot, forecast_plot, f"検出された異常: {len(anomalies)}件", anomalies_df, signals_table, agent_findings, df, anomalies
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(error_traceback)
        return None, None, f"エラーが発生しました: {str(e)}", pd.DataFrame(), pd.DataFrame(), {}, None, None

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
                        
                        # クロスチェックエージェントの場合、追加指標も表示
                        if agent_name == "クロスチェックエージェント" and "additional_metrics" in data:
                            html += "<h5>追加指標:</h5>"
                            html += f"<pre style='white-space: pre-wrap;'>{data['additional_metrics']}</pre>"
                        
                        html += "<hr>"
            
            html += "</div>"
            html += "</details>"
    
    return html

# 評価関数
def run_evaluation(df, anomalies, known_anomalies_str, delay_tolerance):
    try:
        # 既知の異常を解析
        known_anomalies = [date.strip() for date in known_anomalies_str.split(',') if date.strip()]
        
        # 評価器を初期化
        evaluator = AnomalyEvaluator(known_anomalies)
        
        # 評価を実行
        eval_results = evaluator.evaluate(df, anomalies, int(delay_tolerance))
        
        # 評価指標をDataFrameに変換
        metrics_df = pd.DataFrame({
            '指標': ['適合率', '再現率', 'F1スコア', 'Fβスコア', '検知遅延'],
            '値': [
                f"{eval_results['precision']:.4f}",
                f"{eval_results['recall']:.4f}",
                f"{eval_results['f1_score']:.4f}",
                f"{eval_results['f_beta_score']:.4f}",
                f"{eval_results['detection_delay'] if eval_results['detection_delay'] is not None else 'N/A'}"
            ]
        })
        
        # 混同行列を作成
        cm_df = pd.DataFrame({
            '': ['実際: 正常', '実際: 異常'],
            '検出: 正常': [eval_results['true_negatives'], eval_results['false_negatives']],
            '検出: 異常': [eval_results['false_positives'], eval_results['true_positives']]
        })
        
        return metrics_df, cm_df
    
    except Exception as e:
        import traceback
        error_df = pd.DataFrame({'エラー': [str(e), traceback.format_exc()]})
        return error_df, pd.DataFrame()

# 手法比較関数
def compare_methods(df, methods, thresholds_text, known_anomalies_str):
    try:
        # 閾値を解析
        thresholds = [float(t.strip()) for t in thresholds_text.split(',') if t.strip()]
        
        # 既知の異常を解析
        known_anomalies = [date.strip() for date in known_anomalies_str.split(',') if date.strip()]
        
        # 評価器を初期化
        evaluator = AnomalyEvaluator(known_anomalies)
        
        # 比較を実行
        results_df = evaluator.compare_methods(df, methods, thresholds)
        
        return results_df
    
    except Exception as e:
        import traceback
        error_df = pd.DataFrame({'エラー': [str(e), traceback.format_exc()]})
        return error_df

# Gradio UIの作成
def create_gradio_ui():
    with gr.Blocks(title="異常検知分析システム") as app:
        gr.Markdown("# マルチエージェントLLM異常検知分析システム")
        gr.Markdown("時系列データの異常を検出し、LLMエージェントが分析します。")
        
        # データおよび結果を保持する状態
        stored_df = gr.State(None)
        stored_anomalies = gr.State(None)
        
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
                    include_extra_indicators = gr.Checkbox(
                        label="追加指標を含める（出来高、VIX、ドル円）",
                        value=True,
                        info="サンプルデータに追加指標を含めるか選択してください"
                    )
                    
                    # データソースが変更されたときのイベントハンドラ
                    def update_file_visibility(choice):
                        return gr.update(visible=(choice == "upload"))
                    
                    data_source.change(fn=update_file_visibility, inputs=data_source, outputs=file_path)
                    
                with gr.Column():
                    gr.Markdown("### 異常検知設定")
                    detection_method = gr.Dropdown(
                        label="検出方法",
                        choices=["z_score", "iqr", "moving_avg", "isolation_forest", "deep_svdd"],
                        value="z_score",
                        info="Z-score: 標準偏差ベース, IQR: 四分位数ベース, 移動平均: トレンドからの乖離, Isolation Forest: 孤立森, Deep SVDD: ディープサポートベクターデータ記述"
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
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 予測設定")
                    generate_signals = gr.Checkbox(
                        label="売買シグナルと将来予測を生成",
                        value=True,
                        info="異常から売買シグナルを生成し、将来価格を予測します"
                    )
                    forecast_days = gr.Slider(
                        label="予測日数",
                        minimum=10,
                        maximum=90,
                        value=30,
                        step=5,
                        info="将来の何日分を予測するか"
                    )
                    forecast_model = gr.Radio(
                        label="予測モデル",
                        choices=["lstm", "timesfm"],
                        value=FORECASTING_MODEL,
                        info="LSTM: 長短期記憶ネットワーク, TimesFM: Transformerベースモデル"
                    )
            
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
                
                # 予測プロットタブ
                with gr.TabItem("予測"):
                    forecast_plot_output = gr.Plot()
                
                # 異常テーブルタブ
                with gr.TabItem("検出された異常"):
                    anomaly_count = gr.Markdown("")
                    anomaly_table = gr.DataFrame()
                
                # シグナルテーブルタブ
                with gr.TabItem("売買シグナル"):
                    signals_table_text = gr.Markdown("異常から生成された売買シグナル")
                    signals_table = gr.DataFrame()
                
                # エージェント分析タブ
                with gr.TabItem("エージェント分析"):
                    agent_results = gr.HTML("")
                
                # 評価タブを追加
                with gr.TabItem("評価"):
                    gr.Markdown("### 異常検知の精度評価")
                    gr.Markdown("既知の異常日付を入力して、検出精度を評価します。")
                    
                    with gr.Row():
                        known_anomalies = gr.Textbox(
                            label="既知の異常日付（カンマ区切り、YYYY-MM-DD形式）",
                            placeholder="例: 1987-10-19, 2008-10-13, 2020-03-16",
                            value="1987-10-19, 2008-10-13, 2020-03-16"  # デフォルト値
                        )
                        delay_tolerance = gr.Number(
                            label="検知遅延許容値（日数）",
                            value=0,
                            minimum=0,
                            step=1,
                            info="検知が何日遅れても正解とみなすか"
                        )
                    
                    evaluate_btn = gr.Button("評価実行", variant="primary")
                    
                    with gr.Accordion("評価指標の解説", open=False):
                        gr.Markdown("""
                        ### 評価指標について
                        
                        - **適合率（Precision）**: 検出された異常のうち、実際に異常だった割合。値が高いほど誤検知（正常なのに異常と判定）が少ない。
                        - **再現率（Recall）**: 実際の異常のうち、検出できた割合。値が高いほど見逃しが少ない。
                        - **F1スコア**: 適合率と再現率の調和平均。総合的な性能指標。
                        - **Fβスコア**: 再現率をより重視した指標（β=2）。重大な異常を見逃したくない場合に有用。
                        - **検知遅延**: 実際の異常発生から検出までの平均遅れ時間（日数）。
                        """)
                    
                    gr.Markdown("#### 基本評価指標")
                    evaluation_metrics = gr.DataFrame()
                    
                    gr.Markdown("#### 混同行列")
                    confusion_matrix_df = gr.DataFrame()
                    
                    gr.Markdown("#### 複数手法・閾値の比較")
                    with gr.Row():
                        methods_select = gr.CheckboxGroup(
                            label="検出手法",
                            choices=["z_score", "iqr", "moving_avg", "isolation_forest", "deep_svdd"],
                            value=["z_score", "iqr", "moving_avg"]
                        )
                        thresholds_text = gr.Textbox(
                            label="閾値（カンマ区切り）",
                            value="2.0, 2.5, 3.0, 3.5",
                            placeholder="例: 2.0, 2.5, 3.0, 3.5"
                        )
                    
                    compare_btn = gr.Button("手法比較", variant="secondary")
                    comparison_results = gr.DataFrame()
        
        # 実行前の状態表示
        def set_running_status():
            return "分析を実行中です。結果が表示されるまでお待ちください..."
        
        # 分析実行後の結果表示と状態保存
        def show_results_and_store(plot, forecast_plot, count, anomalies_table, signals_table, findings, df, anomalies):
            agent_html = format_agent_findings(findings)
            return (
                "",  # ステータステキストをクリア
                plot,  # プロット
                forecast_plot,  # 予測プロット
                count,  # 異常カウント
                anomalies_table,  # 異常テーブル
                signals_table,  # シグナルテーブル
                agent_html,  # エージェント分析結果
                df,  # データフレームを状態に保存
                anomalies  # 異常を状態に保存
            )
        
        # 評価ボタンイベント
        def handle_evaluate(stored_df, stored_anomalies, known_anomalies_str, delay_tolerance):
            if stored_df is None or stored_anomalies is None:
                return pd.DataFrame({'エラー': ['先に異常検知を実行してください']}), pd.DataFrame()
            
            return run_evaluation(stored_df, stored_anomalies, known_anomalies_str, delay_tolerance)
        
        # 手法比較ボタンイベント
        def handle_compare(stored_df, methods, thresholds_text, known_anomalies_str):
            if stored_df is None:
                return pd.DataFrame({'エラー': ['先に異常検知を実行してください']})
            
            return compare_methods(stored_df, methods, thresholds_text, known_anomalies_str)
        
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
                use_web_agent, use_knowledge_agent, use_crosscheck_agent, use_report_agent, use_manager_agent,
                include_extra_indicators, generate_signals, forecast_days
            ],
            outputs=[plot_output, forecast_plot_output, anomaly_count, anomaly_table, signals_table, agent_results, stored_df, stored_anomalies]
        ).then(
            fn=show_results_and_store,
            inputs=[plot_output, forecast_plot_output, anomaly_count, anomaly_table, signals_table, agent_results, stored_df, stored_anomalies],
            outputs=[status_text, plot_output, forecast_plot_output, anomaly_count, anomaly_table, signals_table, agent_results, stored_df, stored_anomalies]
        )
        
        # 評価ボタンイベント
        evaluate_btn.click(
            fn=handle_evaluate,
            inputs=[stored_df, stored_anomalies, known_anomalies, delay_tolerance],
            outputs=[evaluation_metrics, confusion_matrix_df]
        )
        
        # 手法比較ボタンイベント
        compare_btn.click(
            fn=handle_compare,
            inputs=[stored_df, methods_select, thresholds_text, known_anomalies],
            outputs=comparison_results
        )
        
        # モデル変更イベント
        def update_forecasting_model(model):
            global FORECASTING_MODEL
            FORECASTING_MODEL = model
            return None
        
        forecast_model.change(fn=update_forecasting_model, inputs=[forecast_model], outputs=[])
    
    return app

# メイン処理
if __name__ == "__main__":
    app = create_gradio_ui()
    app.queue().launch(share=False)
