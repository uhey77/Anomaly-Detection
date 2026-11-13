import logging

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
    progress=None,
):
    try:
        if progress is None:
            def progress(fraction, desc=""):
                return None
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



# Streamlit UI: analysis functions above are deliberately framework-independent.
def create_streamlit_ui():
    """Run the six-tab financial anomaly dashboard in Streamlit."""
    import tempfile
    from pathlib import Path

    import streamlit as st

    st.set_page_config(page_title="マルチエージェントLLM異常検知", page_icon="📈", layout="wide")
    st.title("マルチエージェントLLM異常検知分析システム")
    st.caption("時系列異常検知・金融市場分析・将来予測")

    method_labels = {
        "Z-Score（標準偏差ベース）": "z_score",
        "IQR（四分位数ベース）": "iqr",
        "移動平均（トレンド乖離）": "moving_avg",
        "Isolation Forest（機械学習）": "isolation_forest",
        "Deep SVDD（深層学習）": "deep_svdd",
    }
    llm_labels = {"OpenAI GPT": "openai", "Hugging Face": "huggingface", "Mock（デモ用）": "mock"}

    with st.form("analysis_config"):
        st.subheader("分析設定")
        data_col, detection_col = st.columns(2)
        with data_col:
            source = st.radio("データソース", ["サンプルデータを使用", "ファイルをアップロード"])
            uploaded_file = (
                st.file_uploader("時系列データファイル (CSV/XLSX)", type=["csv", "xlsx"])
                if source == "ファイルをアップロード"
                else None
            )
            extra_indicators = st.checkbox("追加指標を含める（出来高、VIX、ドル円）", value=True)
        with detection_col:
            method_label = st.selectbox("検出方法", list(method_labels))
            threshold = st.slider("検出閾値", 1.0, 5.0, 3.0, 0.1)

        llm_col, agent_col = st.columns(2)
        with llm_col:
            llm_label = st.radio("LLMプロバイダ", list(llm_labels), index=2)
            signals_enabled = st.checkbox("売買シグナルと将来予測を生成", value=True)
            forecast_days = st.slider("予測日数", 10, 90, 30, 5)
        with agent_col:
            st.markdown("**エージェント設定**")
            use_web = st.checkbox("Web情報エージェント", value=True)
            use_knowledge = st.checkbox("知識ベースエージェント", value=True)
            use_crosscheck = st.checkbox("クロスチェックエージェント", value=True)
            use_report = st.checkbox("レポート統合エージェント", value=True)
            use_manager = st.checkbox("管理者エージェント", value=True)
        analyze = st.form_submit_button("異常検知分析を開始", type="primary")

    if analyze:
        if source == "ファイルをアップロード" and uploaded_file is None:
            st.error("CSVまたはXLSXファイルを選択してください。")
        else:
            temporary_path = None
            try:
                if uploaded_file is not None:
                    extension = Path(uploaded_file.name).suffix.lower()
                    if extension not in {".csv", ".xlsx"}:
                        raise ValueError("CSVまたはXLSXファイルを選択してください。")
                    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temporary:
                        temporary.write(uploaded_file.getvalue())
                        temporary_path = Path(temporary.name)

                with st.status("分析中...", expanded=True) as status:
                    progress_bar = st.progress(0)

                    def update_progress(fraction, desc=""):
                        percentage = max(0, min(100, round(float(fraction) * 100)))
                        progress_bar.progress(percentage, text=desc)

                    results = run_analysis(
                        "sample" if source == "サンプルデータを使用" else "upload",
                        temporary_path,
                        method_labels[method_label],
                        threshold,
                        llm_labels[llm_label],
                        use_web,
                        use_knowledge,
                        use_crosscheck,
                        use_report,
                        use_manager,
                        include_extra_indicators=extra_indicators,
                        generate_signals=signals_enabled,
                        forecast_days=forecast_days,
                        progress=update_progress,
                    )
                    status.update(
                        label="分析完了" if results[7] is not None else "分析失敗",
                        state="complete" if results[7] is not None else "error",
                    )

                if results[7] is not None:
                    st.session_state["analysis_result"] = results
                    st.session_state.pop("evaluation_result", None)
                    st.session_state.pop("comparison_result", None)
                else:
                    st.session_state.pop("analysis_result", None)
                    st.error("分析に失敗しました。ログと入力内容を確認してください。")
            except Exception as error:
                logger.exception("Streamlitからの分析処理に失敗しました")
                st.error(f"分析エラー: {error}")
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    results = st.session_state.get("analysis_result")
    if results is None:
        st.info("設定を選択して「異常検知分析を開始」を押してください。")
        return

    plot, forecast_plot, _, _, anomaly_table, signals_table, findings, df, anomalies = results
    st.success(f"分析が完了しました。{len(anomalies)}件の異常を検出しました。")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("検出された異常数", len(anomalies))
    metric2.metric("総データポイント数", f"{len(df):,}")
    metric3.metric("異常検出率", f"{len(anomalies) / len(df) * 100:.3f}%")

    tabs = st.tabs(["時系列分析", "将来予測", "検出された異常", "投資シグナル", "AI分析結果", "性能評価"])
    with tabs[0]:
        st.plotly_chart(plot, width="stretch")
    with tabs[1]:
        if forecast_plot is not None:
            st.plotly_chart(forecast_plot, width="stretch")
        else:
            st.info("予測は未生成です。異常が検出され、予測が有効な場合に表示されます。")
    with tabs[2]:
        st.dataframe(anomaly_table, width="stretch", hide_index=True)
    with tabs[3]:
        st.dataframe(signals_table, width="stretch", hide_index=True)
    with tabs[4]:
        st.subheader("マルチエージェント総合分析")
        if not findings:
            st.info("エージェント分析結果はありません。")
        else:
            for name, finding in findings.items():
                with st.expander(str(name), expanded=name in {"管理者エージェント", "レポート統合エージェント"}):
                    if isinstance(finding, dict):
                        if "error" in finding:
                            st.error(str(finding["error"]))
                        if "findings" in finding:
                            st.write(finding["findings"])
                    else:
                        st.write(finding)

    with tabs[5]:
        st.subheader("異常検知精度の評価")
        known_dates = st.text_input(
            "既知の異常日付（カンマ区切り・YYYY-MM-DD）",
            value="1987-10-19, 2008-10-13, 2020-03-16",
        )
        delay = st.number_input("検知遅延許容値（日数）", min_value=0, value=0, step=1)
        evaluate_col, compare_col = st.columns(2)
        with evaluate_col:
            if st.button("評価実行"):
                st.session_state["evaluation_result"] = run_evaluation(df, anomalies, known_dates, delay)
        with compare_col:
            if st.button("手法比較"):
                selected = st.session_state.get("comparison_methods", ["z_score", "iqr", "moving_avg"])
                limits = st.session_state.get("comparison_thresholds", "2.0, 2.5, 3.0, 3.5")
                st.session_state["comparison_result"] = compare_methods(df, selected, limits, known_dates)
        with st.expander("評価指標の詳細説明"):
            st.markdown(
                "適合率：検出した異常の正解割合。再現率：実際の異常を検出した割合。"
                "F1：両者の調和平均。Fβ（β=2）：再現率を重視するスコア。"
                "検知遅延：異常発生から検出までの平均日数。"
            )
        if "evaluation_result" in st.session_state:
            metrics, matrix = st.session_state["evaluation_result"]
            st.markdown("**基本評価指標**")
            st.dataframe(metrics, width="stretch", hide_index=True)
            st.markdown("**混同行列**")
            st.dataframe(matrix, width="stretch", hide_index=True)

        st.markdown("**複数手法・閾値の比較**")
        st.multiselect(
            "比較する検出手法",
            options=list(method_labels.values()),
            default=["z_score", "iqr", "moving_avg"],
            key="comparison_methods",
        )
        st.text_input("比較する閾値（カンマ区切り）", "2.0, 2.5, 3.0, 3.5", key="comparison_thresholds")
        if "comparison_result" in st.session_state:
            st.dataframe(st.session_state["comparison_result"], width="stretch", hide_index=True)


if __name__ == "__main__":
    create_streamlit_ui()
