import logging

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agents.crosscheck_agent import CrossCheckAgent
from agents.knowledge_agent import KnowledgeBaseAgent
from agents.manager_agent import ManagerAgent
from agents.report_agent import ReportIntegrationAgent
from agents.web_agent import WebInformationAgent
from config import (
    ANOMALY_PARAMS,
    FORECASTING_MODEL,
    FORECASTING_PARAMS,
    HF_API_KEY,
    HF_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SIGNAL_THRESHOLD,
    SIGNAL_WINDOW,
)
from detection import create_detector
from evaluation import AnomalyEvaluator
from models.forecasting_pipeline import ForecastingPipeline
from services.data_service import prepare_data
from ui.styles import CUSTOM_CSS
from utils.llm_clients import HuggingFaceClient, MockLLMClient, OpenAIClient
from utils.signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


def detect_anomalies(df, method, threshold, extra_indicators=None):
    """入力データを正規化し、指定手法で異常を検出する。"""
    try:
        method_params = ANOMALY_PARAMS.get(method, {})

        # データフレームのコピーを作成し、データ型を統一
        df_copy = df.copy()

        if df_copy.empty or len(df_copy) < 5:
            logger.warning("データが不足しています（%d件）", len(df_copy))
            return pd.DataFrame()

        # 数値カラムをfloat64に統一
        numeric_columns = ["Close"]
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
                df_copy[col] = df_copy[col].astype(np.float64)

        # NaN値を除去
        original_length = len(df_copy)
        df_copy = df_copy.dropna(subset=["Close"])
        final_length = len(df_copy)

        if df_copy.empty or len(df_copy) < 3:
            logger.warning(
                "有効なデータが不足しています（%d件、元: %d件）",
                final_length,
                original_length,
            )
            return pd.DataFrame()

        logger.info("異常検知実行: 手法=%s データ数=%d", method, len(df_copy))

        extra_features = None
        if extra_indicators:
            extra_features = []
            if "Volume" in df_copy.columns:
                df_copy["Volume"] = pd.to_numeric(df_copy["Volume"], errors="coerce").fillna(0)
                df_copy["Volume"] = df_copy["Volume"].astype(np.float64)
                volume_ma = df_copy["Volume"].rolling(window=20, min_periods=1).mean()
                df_copy["volume_ratio"] = df_copy["Volume"] / (volume_ma + 1e-8)
                df_copy["volume_ratio"] = df_copy["volume_ratio"].astype(np.float64)
                extra_features.append("volume_ratio")

            if "VIX" in df_copy.columns:
                df_copy["VIX"] = pd.to_numeric(df_copy["VIX"], errors="coerce").fillna(20.0)
                df_copy["VIX"] = df_copy["VIX"].astype(np.float64)
                extra_features.append("VIX")

            if "USDJPY" in df_copy.columns:
                df_copy["USDJPY"] = pd.to_numeric(df_copy["USDJPY"], errors="coerce").fillna(110.0)
                df_copy["USDJPY"] = df_copy["USDJPY"].astype(np.float64)
                df_copy["usdjpy_pct_change"] = df_copy["USDJPY"].pct_change() * 100
                df_copy["usdjpy_pct_change"] = df_copy["usdjpy_pct_change"].astype(np.float64)
                extra_features.append("usdjpy_pct_change")

        detector = create_detector(method, threshold, method_params)
        return detector.detect(df_copy, extra_features=extra_features)

    except Exception:
        logger.exception("異常検知でエラーが発生しました")
        return pd.DataFrame()


def create_analysis_plot(df, anomalies):
    """価格、異常、日次変化率をまとめたプロットを作成する。"""
    try:
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("価格データと検出された異常", "日次変化率"),
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3],
        )

        # データ型の安全な変換
        df_safe = df.copy()
        if "Close" in df_safe.columns:
            df_safe["Close"] = pd.to_numeric(df_safe["Close"], errors="coerce")

        # メインの価格データ
        fig.add_trace(
            go.Scatter(
                x=df_safe["Date"],
                y=df_safe["Close"],
                mode="lines",
                name="価格",
                line=dict(color="#2E86AB", width=2),
                hovertemplate="<b>日付:</b> %{x}<br><b>価格:</b> ¥%{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 異常値
        if not anomalies.empty:
            anomalies_safe = anomalies.copy()
            if "Close" in anomalies_safe.columns:
                anomalies_safe["Close"] = pd.to_numeric(anomalies_safe["Close"], errors="coerce")
            if "pct_change" in anomalies_safe.columns:
                anomalies_safe["pct_change"] = pd.to_numeric(
                    anomalies_safe["pct_change"], errors="coerce"
                )
            else:
                anomalies_safe["pct_change"] = 0.0

            fig.add_trace(
                go.Scatter(
                    x=anomalies_safe["Date"],
                    y=anomalies_safe["Close"],
                    mode="markers",
                    name="検出された異常",
                    marker=dict(
                        color="#E63946",
                        size=12,
                        symbol="diamond",
                        line=dict(color="white", width=2),
                    ),
                    hovertemplate="<b>異常日:</b> %{x}<br><b>価格:</b> ¥%{y:,.0f}<br><b>変化率:</b> %{customdata:.2f}%<extra></extra>",
                    customdata=anomalies_safe["pct_change"].fillna(0),
                ),
                row=1,
                col=1,
            )

        # 変化率の計算と表示
        if "pct_change" not in df_safe.columns:
            df_safe["pct_change"] = df_safe["Close"].pct_change() * 100

        df_safe["pct_change"] = pd.to_numeric(df_safe["pct_change"], errors="coerce").fillna(0)

        # 変化率プロット
        colors = ["#E63946" if abs(float(x)) > 2 else "#2E86AB" for x in df_safe["pct_change"]]

        fig.add_trace(
            go.Bar(
                x=df_safe["Date"],
                y=df_safe["pct_change"],
                name="日次変化率",
                marker_color=colors,
                opacity=0.7,
                hovertemplate="<b>日付:</b> %{x}<br><b>変化率:</b> %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=dict(
                text="<b>時系列データ分析結果</b>",
                x=0.5,
                font=dict(size=20, color="#2c3e50"),
            ),
            showlegend=True,
            height=700,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        fig.update_xaxes(title_text="日付", row=2, col=1)
        fig.update_yaxes(title_text="価格", row=1, col=1)
        fig.update_yaxes(title_text="変化率 (%)", row=2, col=1)

        return fig

    except Exception:
        logger.exception("プロット作成中にエラーが発生しました")
        # エラーの場合は空のプロットを返す
        return go.Figure().add_annotation(
            text="プロット作成中にエラーが発生しました", x=0.5, y=0.5, showarrow=False
        )


def create_forecast_plot(df, anomalies, forecast_df, adjusted_forecast_df):
    """実績価格と補正前後の予測をプロットする。"""
    fig = go.Figure()

    # 実際の価格（最近のデータのみ表示）
    recent_data = df.tail(100)
    fig.add_trace(
        go.Scatter(
            x=recent_data["Date"],
            y=recent_data["Close"],
            mode="lines",
            name="実際の価格",
            line=dict(color="#2E86AB", width=3),
            hovertemplate="<b>日付:</b> %{x}<br><b>実際価格:</b> ¥%{y:,.0f}<extra></extra>",
        )
    )

    # 異常値
    if not anomalies.empty:
        recent_anomalies = anomalies[anomalies["Date"] >= recent_data["Date"].min()]
        if not recent_anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=recent_anomalies["Date"],
                    y=recent_anomalies["Close"],
                    mode="markers",
                    name="検出された異常",
                    marker=dict(
                        color="#E63946",
                        size=12,
                        symbol="diamond",
                        line=dict(color="white", width=2),
                    ),
                )
            )

    # 基本予測
    if forecast_df is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["predicted_price"],
                mode="lines",
                name="基本予測",
                line=dict(color="#F77F00", width=2, dash="dash"),
                hovertemplate="<b>予測日:</b> %{x}<br><b>基本予測:</b> ¥%{y:,.0f}<extra></extra>",
            )
        )

    # 異常調整済み予測
    if adjusted_forecast_df is not None:
        fig.add_trace(
            go.Scatter(
                x=adjusted_forecast_df["Date"],
                y=adjusted_forecast_df["predicted_price"],
                mode="lines",
                name="異常調整済み予測",
                line=dict(color="#6A994E", width=3),
                hovertemplate="<b>予測日:</b> %{x}<br><b>調整済み予測:</b> ¥%{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="<b>価格予測分析</b>", x=0.5, font=dict(size=20, color="#2c3e50")),
        xaxis_title="日付",
        yaxis_title="価格",
        template="plotly_white",
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# LLMクライアントを取得
def get_llm_client(provider):
    """指定されたプロバイダに基づいてLLMクライアントを取得"""
    if provider == "openai":
        return OpenAIClient(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    if provider == "huggingface":
        return HuggingFaceClient(api_key=HF_API_KEY, model=HF_MODEL)
    return MockLLMClient()


# エージェントシステムを実行
def run_agent_system(anomalies, llm_provider, enabled_agents, reference_data=None):
    """エージェントシステムを実行して結果を返す"""
    llm_client = get_llm_client(llm_provider)

    # 重要な異常のみに絞る
    if len(anomalies) > 3:
        anomalies = anomalies.iloc[anomalies["pct_change"].abs().argsort()[::-1][:3]].copy()

    agent_classes = {
        "web": WebInformationAgent,
        "knowledge": KnowledgeBaseAgent,
        "crosscheck": CrossCheckAgent,
    }
    agents = [
        agent_class(llm_client=llm_client)
        for name, agent_class in agent_classes.items()
        if name in enabled_agents
    ]

    # コンテキスト準備
    context = {"reference_data": reference_data if reference_data else {}}

    # エージェント結果を保存
    agent_findings = {}

    # 各エージェントを実行
    for agent in agents:
        agent_result = agent.process(anomalies, context)
        agent_findings[agent.name] = agent_result

    # 統合エージェントを実行
    if "report" in enabled_agents:
        report_agent = ReportIntegrationAgent(llm_client=llm_client)
        context["agent_findings"] = agent_findings
        report_agent_result = report_agent.process(anomalies, context)
        agent_findings[report_agent.name] = report_agent_result

    # 管理者エージェントを実行
    if "manager" in enabled_agents:
        manager_agent = ManagerAgent(llm_client=llm_client)
        manager_agent_result = manager_agent.process(anomalies, context)
        agent_findings[manager_agent.name] = manager_agent_result

    return agent_findings


# メトリクスカードのHTML生成
def create_metrics_cards(anomalies_count, total_data_points, detection_rate):
    """メトリクスカードのHTMLを生成"""
    return f"""
    <div class="metrics-grid fade-in">
        <div class="metric-card">
            <div class="metric-value">{anomalies_count}</div>
            <div class="metric-label">検出された異常数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_data_points:,}</div>
            <div class="metric-label">総データポイント数</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{detection_rate:.3f}%</div>
            <div class="metric-label">異常検出率</div>
        </div>
    </div>
    """


# 分析を実行
def run_analysis(
    data_source,
    file_path,
    detection_method,
    threshold,
    llm_provider,
    use_web_agent,
    use_knowledge_agent,
    use_crosscheck_agent,
    use_report_agent,
    use_manager_agent,
    include_extra_indicators=True,
    generate_signals=True,
    forecast_days=30,
    progress=gr.Progress(),
):
    try:
        # プログレス初期化
        progress(0.0, desc="分析を開始しています...")

        # データを準備
        progress(0.125, desc="データを準備中...")
        use_sample = data_source == "sample"
        df = prepare_data(
            use_sample,
            file_path,
            include_extra_indicators=include_extra_indicators,
        )

        # 異常検知を実行
        progress(0.25, desc="異常検知を実行中...")
        anomalies = detect_anomalies(df, detection_method, threshold, include_extra_indicators)

        # プロットを作成
        progress(0.375, desc="可視化を作成中...")
        plot = create_analysis_plot(df, anomalies)

        # 基本的な統計情報
        anomalies_count = len(anomalies)
        total_data_points = len(df)
        detection_rate = (anomalies_count / total_data_points) * 100 if total_data_points > 0 else 0

        # メトリクスカードを作成
        metrics_html = create_metrics_cards(anomalies_count, total_data_points, detection_rate)

        # 異常データを表として整形
        if anomalies.empty:
            anomalies_df = pd.DataFrame(columns=["日付", "値", "変化率 (%)"])
        else:
            anomalies_df = anomalies.copy()
            anomalies_df = anomalies_df[["Date", "Close", "pct_change"]]
            anomalies_df.columns = ["日付", "値", "変化率 (%)"]
            anomalies_df["日付"] = anomalies_df["日付"].dt.strftime("%Y-%m-%d")
            anomalies_df["変化率 (%)"] = anomalies_df["変化率 (%)"].round(2)

        # 有効なエージェントのリストを作成
        enabled_agents = [
            name
            for name, enabled in (
                ("web", use_web_agent),
                ("knowledge", use_knowledge_agent),
                ("crosscheck", use_crosscheck_agent),
                ("report", use_report_agent),
                ("manager", use_manager_agent),
            )
            if enabled
        ]

        # 参照データを準備
        reference_data = {}
        if include_extra_indicators:
            if "Volume" in df.columns:
                volume_df = df[["Date", "Volume"]].copy()
                reference_data["volume"] = volume_df

            if "VIX" in df.columns:
                vix_df = df[["Date", "VIX"]].copy()
                reference_data["vix"] = vix_df

            if "USDJPY" in df.columns:
                usdjpy_df = df[["Date", "USDJPY"]].copy()
                reference_data["usdjpy"] = usdjpy_df

        # エージェント分析を実行
        progress(0.5, desc="エージェント分析を実行中...")

        agent_findings = {}
        if enabled_agents and not anomalies.empty:
            agent_findings = run_agent_system(
                anomalies, llm_provider, enabled_agents, reference_data
            )

        # シグナル生成と予測
        progress(0.75, desc="予測モデルを実行中...")

        signals_df = None
        forecast_df = None
        adjusted_forecast_df = None
        forecast_plot = None

        if generate_signals and not anomalies.empty:
            signal_generator = SignalGenerator(
                threshold=SIGNAL_THRESHOLD, window_size=SIGNAL_WINDOW
            )

            signals_df = signal_generator.generate_signals(df, anomalies)

            # 予測パイプライン
            pipeline = ForecastingPipeline(
                forecasting_model=FORECASTING_MODEL,
                signal_threshold=SIGNAL_THRESHOLD,
                window_size=SIGNAL_WINDOW,
                lookback=FORECASTING_PARAMS[FORECASTING_MODEL].get("lookback", 60),
            )

            # シグナルは上で生成済みなので、価格予測のみ受け取る
            _, forecast_df, _ = pipeline.process(
                df,
                anomalies,
                train_model=True,
                days_ahead=forecast_days,
                adjustment_weight=0.5,
            )

            # 予測補正を手動実行
            adjusted_forecast_df = signal_generator.forecast_adjustment(
                signals_df, forecast_df, weight=0.5
            )

            forecast_plot = create_forecast_plot(df, anomalies, forecast_df, adjusted_forecast_df)

        # シグナルを表として整形
        signals_table = pd.DataFrame()
        if signals_df is not None and not signals_df.empty:
            signals_table = signals_df[signals_df["signal"] != 0][
                ["Date", "Close", "pct_change", "signal", "anomaly_score"]
            ]
            if not signals_table.empty:
                signals_table.columns = [
                    "日付",
                    "価格",
                    "変化率 (%)",
                    "シグナル(-1=売/1=買)",
                    "異常強度",
                ]
                signals_table["日付"] = signals_table["日付"].dt.strftime("%Y-%m-%d")
                signals_table["変化率 (%)"] = signals_table["変化率 (%)"].round(2)
                signals_table["異常強度"] = signals_table["異常強度"].round(2)

        # 最終更新
        progress(1.0, desc="分析完了!")

        # 成功メッセージ
        status_html = f'<div class="status-success fade-in">✅ 分析が正常に完了しました！{anomalies_count}件の異常を検出しました。</div>'

        return (
            plot,
            forecast_plot,
            status_html,
            metrics_html,
            anomalies_df,
            signals_table,
            agent_findings,
            df,
            anomalies,
        )

    except Exception as error:
        logger.exception("分析中にエラーが発生しました")

        error_html = f'<div class="status-error fade-in">❌ エラーが発生しました: {error}</div>'
        empty_metrics = '<div class="metrics-grid"></div>'

        return (
            None,
            None,
            error_html,
            empty_metrics,
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            None,
            None,
        )


def format_agent_findings(agent_findings):
    """エージェント結果を画面表示用HTMLに整形する。"""
    # agent_findingsが空辞書、None、または空のDataFrameかどうかを安全にチェック
    if (
        agent_findings is None
        or (isinstance(agent_findings, dict) and len(agent_findings) == 0)
        or (hasattr(agent_findings, "empty") and agent_findings.empty)
    ):
        return '<div class="status-warning fade-in">⚠️ エージェント分析結果はありません。</div>'

    html = '<div class="fade-in">'

    # 最終評価が存在する場合はそれを優先表示
    if "管理者エージェント" in agent_findings:
        manager_findings = agent_findings["管理者エージェント"].get("findings", {})
        if manager_findings:
            html += """
            <div class="agent-result">
                <div class="agent-header">
                    <h3>🎯 最終評価・推奨事項</h3>
                    <span>⭐ エグゼクティブサマリー</span>
                </div>
                <div class="agent-content">
            """

            for date, data in manager_findings.items():
                anomaly_details = data.get("anomaly_details", {})
                pct_change = anomaly_details.get("pct_change", "N/A")

                html += f"""
                <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; border-left: 4px solid #28a745;">
                    <h4 style="color: #155724; margin-bottom: 15px;">📅 {date} (変化率: {pct_change}%)</h4>
                    <pre style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; margin: 0;">{data.get("final_assessment", "N/A")}</pre>
                </div>
                """

            html += "</div></div>"

    # 統合レポートが存在する場合はそれを表示
    if "レポート統合エージェント" in agent_findings:
        report_findings = agent_findings["レポート統合エージェント"].get("findings", {})
        if report_findings:
            html += """
            <div class="agent-result">
                <div class="agent-header">
                    <h3>📋 統合分析レポート</h3>
                    <span>📊 複数エージェント統合結果</span>
                </div>
                <div class="agent-content">
            """

            for date, data in report_findings.items():
                anomaly_details = data.get("anomaly_details", {})
                pct_change = anomaly_details.get("pct_change", "N/A")

                html += f"""
                <div style="margin-bottom: 25px; padding: 20px; background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border-radius: 12px; border-left: 4px solid #856404;">
                    <h4 style="color: #856404; margin-bottom: 15px;">📅 {date} (変化率: {pct_change}%)</h4>
                    <pre style="background: white; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; margin: 0;">{data.get("integrated_report", "N/A")}</pre>
                </div>
                """

            html += "</div></div>"

    # 個別のエージェント結果も表示
    html += '<h3 style="margin: 30px 0 20px 0; color: #495057;">🔍 個別エージェント分析詳細</h3>'

    agent_colors = {
        "Web情報エージェント": "#007bff",
        "知識ベースエージェント": "#28a745",
        "クロスチェックエージェント": "#ffc107",
    }

    agent_icons = {
        "Web情報エージェント": "🌐",
        "知識ベースエージェント": "📚",
        "クロスチェックエージェント": "🔄",
    }

    for agent_name, findings in agent_findings.items():
        if agent_name not in ["レポート統合エージェント", "管理者エージェント"]:
            color = agent_colors.get(agent_name, "#6c757d")
            icon = agent_icons.get(agent_name, "🤖")

            html += f"""
            <details class="agent-result" style="margin-bottom: 20px;">
                <summary class="agent-header" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                    <span>{icon} {agent_name}</span>
                    <span style="font-size: 0.9em; opacity: 0.9;">クリックして詳細を表示</span>
                </summary>
                <div class="agent-content">
            """

            if "error" in findings:
                html += f'<div class="status-error">❌ エラー: {findings["error"]}</div>'
            elif "findings" in findings:
                for date, data in findings["findings"].items():
                    if "llm_analysis" in data:
                        html += f"""
                        <div style="margin-bottom: 20px;">
                            <h4 style="color: {color}; margin-bottom: 10px;">📅 {date}</h4>
                            <pre style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 3px solid {color}; margin-bottom: 15px;">{data["llm_analysis"]}</pre>
                        """

                        # クロスチェックエージェントの場合、追加指標も表示
                        if (
                            agent_name == "クロスチェックエージェント"
                            and "additional_metrics" in data
                        ):
                            html += f"""
                            <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin-top: 10px;">
                                <h5 style="color: #0056b3; margin-bottom: 10px;">📈 追加市場指標</h5>
                                <pre style="background: white; padding: 12px; border-radius: 6px; margin: 0; font-size: 0.9em;">{data["additional_metrics"]}</pre>
                            </div>
                            """

                        html += '</div><hr style="border: none; border-top: 1px solid #dee2e6; margin: 20px 0;">'

            html += "</div></details>"

    html += "</div>"
    return html


# 評価関数
def run_evaluation(df, anomalies, known_anomalies_str, delay_tolerance):
    try:
        known_anomalies = [date.strip() for date in known_anomalies_str.split(",") if date.strip()]

        evaluator = AnomalyEvaluator(known_anomalies)
        eval_results = evaluator.evaluate(df, anomalies, int(delay_tolerance))

        # 評価指標をDataFrameに変換
        metrics_df = pd.DataFrame(
            {
                "評価指標": [
                    "適合率 (Precision)",
                    "再現率 (Recall)",
                    "F1スコア",
                    "Fβスコア",
                    "検知遅延 (日)",
                ],
                "値": [
                    f"{eval_results['precision']:.4f}",
                    f"{eval_results['recall']:.4f}",
                    f"{eval_results['f1_score']:.4f}",
                    f"{eval_results['f_beta_score']:.4f}",
                    f"{eval_results['detection_delay'] if eval_results['detection_delay'] is not None else 'N/A'}",
                ],
                "説明": [
                    "検出された異常のうち、実際に異常だった割合",
                    "実際の異常のうち、検出できた割合",
                    "適合率と再現率の調和平均（総合性能指標）",
                    "再現率を重視した総合指標（重要な異常の見逃し防止）",
                    "実際の異常発生から検出までの平均遅れ時間",
                ],
            }
        )

        # 混同行列を作成
        cm_df = pd.DataFrame(
            {
                "実際→": ["正常", "異常"],
                "予測: 正常": [
                    eval_results["true_negatives"],
                    eval_results["false_negatives"],
                ],
                "予測: 異常": [
                    eval_results["false_positives"],
                    eval_results["true_positives"],
                ],
            }
        )

        return metrics_df, cm_df

    except Exception as error:
        logger.exception("評価中にエラーが発生しました")
        error_df = pd.DataFrame({"エラー": [str(error)]})
        return error_df, pd.DataFrame()


def compare_methods(df, methods, thresholds_text, known_anomalies_str):
    try:
        thresholds = [float(t.strip()) for t in thresholds_text.split(",") if t.strip()]
        known_anomalies = [date.strip() for date in known_anomalies_str.split(",") if date.strip()]

        evaluator = AnomalyEvaluator(known_anomalies)

        results = []

        for method in methods:
            for threshold in thresholds:
                try:
                    # データの前処理とチェック
                    df_processed = df.copy()

                    if df_processed.empty or len(df_processed) < 10:
                        logger.warning(
                            "データ不足のためスキップします: method=%s threshold=%s rows=%d",
                            method,
                            threshold,
                            len(df_processed),
                        )
                        continue

                    # 数値カラムの確認
                    if "Close" not in df_processed.columns:
                        logger.warning(
                            "Closeカラムがないためスキップします: method=%s threshold=%s",
                            method,
                            threshold,
                        )
                        continue

                    # NaN値の処理
                    df_processed["Close"] = pd.to_numeric(df_processed["Close"], errors="coerce")
                    original_length = len(df_processed)
                    df_processed = df_processed.dropna(subset=["Close"])
                    final_length = len(df_processed)

                    if final_length < 5:
                        logger.warning(
                            "有効データ不足のためスキップします: method=%s threshold=%s rows=%d/%d",
                            method,
                            threshold,
                            final_length,
                            original_length,
                        )
                        continue

                    # 機械学習手法に必要なデータ量と変動性を確認する
                    if method in ["isolation_forest", "deep_svdd"]:
                        if len(df_processed) < 30:
                            logger.warning(
                                "機械学習手法には30件以上必要です: method=%s rows=%d",
                                method,
                                len(df_processed),
                            )
                            continue

                        # 変動性のチェック
                        if df_processed["Close"].std() == 0:
                            logger.warning(
                                "価格変動がないためスキップします: method=%s threshold=%s",
                                method,
                                threshold,
                            )
                            continue

                    comparison_params = dict(ANOMALY_PARAMS.get(method, {}))
                    if method == "isolation_forest":
                        comparison_params["n_estimators"] = 50
                    elif method == "deep_svdd":
                        comparison_params.update(epochs=10, batch_size=16)
                    detector = create_detector(method, threshold, comparison_params)
                    detected_anomalies = detector.detect(df_processed)

                    # 評価を実行
                    eval_result = evaluator.evaluate(df_processed, detected_anomalies)

                    # 結果を格納
                    results.append(
                        {
                            "検出手法": method,
                            "閾値": threshold,
                            "適合率": round(eval_result["precision"], 4),
                            "再現率": round(eval_result["recall"], 4),
                            "F1スコア": round(eval_result["f1_score"], 4),
                            "Fβスコア": round(eval_result["f_beta_score"], 4),
                            "検知数": len(detected_anomalies),
                            "真陽性": eval_result["true_positives"],
                            "偽陽性": eval_result["false_positives"],
                            "偽陰性": eval_result["false_negatives"],
                        }
                    )

                except Exception:
                    logger.exception(
                        "手法比較に失敗しました: method=%s threshold=%s",
                        method,
                        threshold,
                    )
                    results.append(
                        {
                            "検出手法": method,
                            "閾値": threshold,
                            "適合率": "エラー",
                            "再現率": "エラー",
                            "F1スコア": "エラー",
                            "Fβスコア": "エラー",
                            "検知数": "エラー",
                            "真陽性": "エラー",
                            "偽陽性": "エラー",
                            "偽陰性": "エラー",
                        }
                    )

        # 結果をDataFrameに変換
        if results:
            results_df = pd.DataFrame(results)
            return results_df
        else:
            return pd.DataFrame(
                {"メッセージ": ["すべての手法でエラーが発生しました。データを確認してください。"]}
            )

    except Exception as error:
        logger.exception("手法比較中にエラーが発生しました")
        error_df = pd.DataFrame(
            {
                "エラー": ["手法比較中にエラーが発生しました"],
                "詳細": [str(error)],
                "推奨対処": ["データ形式を確認するか、異なる手法を試してください"],
            }
        )
        return error_df


# Gradio UIの作成
def create_gradio_ui():
    with gr.Blocks(title="🔍 マルチエージェントLLM異常検知分析システム") as app:
        # ヘッダー
        gr.HTML("""
        <div class="header fade-in">
            <h1>🔍 マルチエージェントLLM異常検知分析システム</h1>
            <p>🤖 AI駆動の時系列データ異常検知と包括的分析プラットフォーム</p>
        </div>
        """)

        # データおよび結果を保持する状態
        stored_df = gr.State(None)
        stored_anomalies = gr.State(None)

        # 設定セクション
        with gr.Group():
            gr.HTML('<div class="config-card fade-in">')
            gr.HTML("<h3>⚙️ 分析設定</h3>")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h4>📊 データ設定</h4>")
                    data_source = gr.Radio(
                        label="📁 データソース",
                        choices=[
                            ("サンプルデータを使用", "sample"),
                            ("ファイルをアップロード", "upload"),
                        ],
                        value="sample",
                        info="🎯 サンプルデータには歴史的な市場異常が含まれています",
                    )
                    file_path = gr.File(
                        label="📋 時系列データファイル (CSV/XLSX)",
                        type="filepath",
                        file_types=[".csv", ".xlsx"],
                        visible=False,
                    )
                    include_extra_indicators = gr.Checkbox(
                        label="📈 追加指標を含める（出来高、VIX、ドル円）",
                        value=True,
                        info="✨ より詳細な市場分析のための追加データ",
                    )

                with gr.Column(scale=1):
                    gr.HTML("<h4>🎯 異常検知設定</h4>")
                    detection_method = gr.Dropdown(
                        label="🔍 検出方法",
                        choices=[
                            ("Z-Score（標準偏差ベース）", "z_score"),
                            ("IQR（四分位数ベース）", "iqr"),
                            ("移動平均（トレンド乖離）", "moving_avg"),
                            ("Isolation Forest（機械学習）", "isolation_forest"),
                            ("Deep SVDD（深層学習）", "deep_svdd"),
                        ],
                        value="z_score",
                        info="📊 各手法の特徴を理解して選択してください",
                    )
                    threshold = gr.Slider(
                        label="🎚️ 検出閾値",
                        minimum=1.0,
                        maximum=5.0,
                        value=3.0,
                        step=0.1,
                        info="⚡ 値が大きいほど検出される異常が少なくなります",
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h4>🤖 LLM設定</h4>")
                    llm_provider = gr.Radio(
                        label="🧠 LLMプロバイダ",
                        choices=[
                            ("OpenAI GPT", "openai"),
                            ("Hugging Face", "huggingface"),
                            ("Mock（デモ用）", "mock"),
                        ],
                        value="mock",
                        info="🔑 OpenAI/HuggingFaceを使用する場合はAPIキーが必要です",
                    )

                with gr.Column(scale=1):
                    gr.HTML("<h4>👥 エージェント設定</h4>")
                    with gr.Row():
                        with gr.Column():
                            use_web_agent = gr.Checkbox(label="🌐 Web情報エージェント", value=True)
                            use_knowledge_agent = gr.Checkbox(
                                label="📚 知識ベースエージェント", value=True
                            )
                            use_crosscheck_agent = gr.Checkbox(
                                label="🔄 クロスチェックエージェント", value=True
                            )
                        with gr.Column():
                            use_report_agent = gr.Checkbox(
                                label="📋 レポート統合エージェント", value=True
                            )
                            use_manager_agent = gr.Checkbox(
                                label="🎯 管理者エージェント", value=True
                            )

            with gr.Row():
                with gr.Column():
                    generate_signals = gr.Checkbox(
                        label="📈 売買シグナルと将来予測を生成",
                        value=True,
                        info="🎯 異常から投資シグナルを生成し、価格を予測します",
                    )
                    forecast_days = gr.Slider(
                        label="📅 予測日数",
                        minimum=10,
                        maximum=90,
                        value=30,
                        step=5,
                        info="🔮 将来の何日分を予測するか",
                    )

            # データソース変更時のイベントハンドラ
            def update_file_visibility(choice):
                return gr.update(visible=(choice == "upload"))

            data_source.change(fn=update_file_visibility, inputs=data_source, outputs=file_path)

            gr.HTML("</div>")

        # 分析実行ボタン
        gr.HTML('<div class="button-container">')
        analyze_btn = gr.Button(
            "🚀 異常検知分析を開始",
            variant="primary",
            size="lg",
            elem_classes=["primary-btn"],
        )
        gr.HTML("</div>")

        # ステータス表示
        status_display = gr.HTML("")
        metrics_display = gr.HTML("")

        # 結果表示セクション
        with gr.Group():
            gr.HTML('<div class="results-card fade-in">')
            gr.HTML("<h3>📊 分析結果</h3>")

            with gr.Tabs():
                # メインプロットタブ
                with gr.TabItem("📈 時系列分析", id="main_plot"):
                    plot_output = gr.Plot(label="時系列データと異常検知結果")

                # 予測プロットタブ
                with gr.TabItem("🔮 将来予測", id="forecast"):
                    forecast_plot_output = gr.Plot(label="価格予測と異常調整")

                # 検出結果タブ
                with gr.TabItem("🎯 検出された異常", id="anomalies"):
                    gr.HTML("<h4>📋 異常検知結果一覧</h4>")
                    anomaly_table = gr.DataFrame(
                        label="検出された異常データ", interactive=False, wrap=True
                    )

                # シグナルタブ
                with gr.TabItem("💰 投資シグナル", id="signals"):
                    gr.HTML("<h4>📊 売買推奨シグナル</h4>")
                    signals_table = gr.DataFrame(
                        label="異常から生成された売買シグナル",
                        interactive=False,
                        wrap=True,
                    )

                # エージェント分析タブ
                with gr.TabItem("🤖 AI分析結果", id="agents"):
                    gr.HTML("<h4>🧠 マルチエージェント総合分析</h4>")
                    agent_results = gr.HTML("")

                # 評価タブ
                with gr.TabItem("📊 性能評価", id="evaluation"):
                    gr.HTML(
                        '<div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">'
                    )
                    gr.HTML("<h4>🎯 異常検知精度の評価</h4>")
                    gr.HTML(
                        "<p>📈 既知の異常日付を入力して、検出アルゴリズムの性能を定量的に評価します。</p>"
                    )
                    gr.HTML("</div>")

                    with gr.Row():
                        with gr.Column():
                            known_anomalies = gr.Textbox(
                                label="📅 既知の異常日付（カンマ区切り、YYYY-MM-DD形式）",
                                placeholder="例: 1987-10-19, 2008-10-13, 2020-03-16",
                                value="1987-10-19, 2008-10-13, 2020-03-16",
                                info="🎯 評価用の正解データとして使用されます",
                            )
                        with gr.Column():
                            delay_tolerance = gr.Number(
                                label="⏱️ 検知遅延許容値（日数）",
                                value=0,
                                minimum=0,
                                step=1,
                                info="🔍 検知が何日遅れても正解とみなすか",
                            )

                    with gr.Row():
                        evaluate_btn = gr.Button(
                            "📊 評価実行",
                            variant="secondary",
                            elem_classes=["secondary-btn"],
                        )
                        compare_btn = gr.Button(
                            "⚖️ 手法比較",
                            variant="secondary",
                            elem_classes=["secondary-btn"],
                        )

                    with gr.Accordion("📚 評価指標の詳細説明", open=False):
                        gr.HTML("""
                        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;">
                            <h5>📊 性能指標について</h5>
                            <ul style="line-height: 1.8;">
                                <li><strong>🎯 適合率（Precision）:</strong> 検出された異常のうち、実際に異常だった割合。値が高いほど誤検知が少ない。</li>
                                <li><strong>🔍 再現率（Recall）:</strong> 実際の異常のうち、検出できた割合。値が高いほど見逃しが少ない。</li>
                                <li><strong>⚖️ F1スコア:</strong> 適合率と再現率の調和平均。総合的な性能指標。</li>
                                <li><strong>📈 Fβスコア:</strong> 再現率をより重視した指標（β=2）。重大な異常を見逃したくない場合に有用。</li>
                                <li><strong>⏱️ 検知遅延:</strong> 実際の異常発生から検出までの平均遅れ時間（日数）。</li>
                            </ul>
                        </div>
                        """)

                    gr.HTML("<h5>📈 基本評価指標</h5>")
                    evaluation_metrics = gr.DataFrame(
                        label="性能評価結果", interactive=False, wrap=True
                    )

                    gr.HTML("<h5>📊 混同行列（予測精度の詳細）</h5>")
                    confusion_matrix_df = gr.DataFrame(label="混同行列", interactive=False)

                    gr.HTML("<h5>⚖️ 複数手法・閾値の比較分析</h5>")
                    with gr.Row():
                        methods_select = gr.CheckboxGroup(
                            label="🔍 比較する検出手法",
                            choices=[
                                ("Z-Score", "z_score"),
                                ("IQR", "iqr"),
                                ("移動平均", "moving_avg"),
                                ("Isolation Forest", "isolation_forest"),
                                ("Deep SVDD", "deep_svdd"),
                            ],
                            value=["z_score", "iqr", "moving_avg"],
                            info="📋 複数の手法を選択して性能を比較できます",
                        )
                        thresholds_text = gr.Textbox(
                            label="🎚️ 比較する閾値（カンマ区切り）",
                            value="2.0, 2.5, 3.0, 3.5",
                            placeholder="例: 2.0, 2.5, 3.0, 3.5",
                            info="📊 異なる閾値での性能を比較します",
                        )

                    comparison_results = gr.DataFrame(
                        label="手法比較結果", interactive=False, wrap=True
                    )

            gr.HTML("</div>")

        # イベントハンドラー
        def handle_results_display(
            plot,
            forecast_plot,
            status,
            metrics,
            anomalies_table,
            signals_table,
            findings,
            df,
            anomalies,
        ):
            agent_html = format_agent_findings(findings)
            return (
                plot,
                forecast_plot,
                status,
                metrics,
                anomalies_table,
                signals_table,
                agent_html,
                df,
                anomalies,
            )

        def handle_evaluate(stored_df, stored_anomalies, known_anomalies_str, delay_tolerance):
            if (
                stored_df is None
                or stored_anomalies is None
                or (hasattr(stored_anomalies, "empty") and stored_anomalies.empty)
            ):
                return pd.DataFrame(
                    {"⚠️ 注意": ["先に異常検知分析を実行してください"]}
                ), pd.DataFrame()

            return run_evaluation(stored_df, stored_anomalies, known_anomalies_str, delay_tolerance)

        def handle_compare(stored_df, methods, thresholds_text, known_anomalies_str):
            if stored_df is None or (hasattr(stored_df, "empty") and stored_df.empty):
                return pd.DataFrame({"⚠️ 注意": ["先に異常検知分析を実行してください"]})
            return compare_methods(stored_df, methods, thresholds_text, known_anomalies_str)

        stored_agent_findings = gr.State(None)

        # 分析実行ボタンのイベント
        analyze_btn.click(
            fn=run_analysis,
            inputs=[
                data_source,
                file_path,
                detection_method,
                threshold,
                llm_provider,
                use_web_agent,
                use_knowledge_agent,
                use_crosscheck_agent,
                use_report_agent,
                use_manager_agent,
                include_extra_indicators,
                generate_signals,
                forecast_days,
            ],
            outputs=[
                plot_output,
                forecast_plot_output,
                status_display,
                metrics_display,
                anomaly_table,
                signals_table,
                stored_agent_findings,  # ← agent_findings をここに
                stored_df,  # ← df
                stored_anomalies,  # ← anomalies
            ],
        ).then(
            fn=handle_results_display,
            inputs=[
                plot_output,
                forecast_plot_output,
                status_display,
                metrics_display,
                anomaly_table,
                signals_table,
                stored_agent_findings,  # findings
                stored_df,  # df
                stored_anomalies,  # anomalies
            ],
            outputs=[
                plot_output,
                forecast_plot_output,
                status_display,
                metrics_display,
                anomaly_table,
                signals_table,
                agent_results,  # HTML へ描画
                stored_df,
                stored_anomalies,
            ],
        )

        # 評価ボタンのイベント
        evaluate_btn.click(
            fn=handle_evaluate,
            inputs=[stored_df, stored_anomalies, known_anomalies, delay_tolerance],
            outputs=[evaluation_metrics, confusion_matrix_df],
        )

        # 手法比較ボタンのイベント
        compare_btn.click(
            fn=handle_compare,
            inputs=[stored_df, methods_select, thresholds_text, known_anomalies],
            outputs=comparison_results,
        )

    return app


# メイン処理
if __name__ == "__main__":
    app = create_gradio_ui()
    app.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green", neutral_hue="gray"),
    )
