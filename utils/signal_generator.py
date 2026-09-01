"""異常検知結果から売買シグナルと予測補正を生成する。"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalGenerator:
    """価格トレンドと異常強度から売買シグナルを生成する。"""

    def __init__(self, threshold: float = 0.5, window_size: int = 5) -> None:
        self.threshold = threshold
        self.window_size = window_size

    def generate_signals(self, data: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
        """異常行を買い・売り・ホールドのシグナルへ変換する。"""
        signals = data.copy()
        if "Close" not in signals:
            raise ValueError("必須カラムがありません: Close")

        signals["Close"] = signals["Close"].astype(np.float64)
        signals["signal"] = 0
        signals["anomaly_score"] = np.float64(0.0)
        if "pct_change" not in signals:
            signals["pct_change"] = signals["Close"].pct_change() * 100
        signals["pct_change"] = signals["pct_change"].astype(np.float64)

        for _, anomaly in anomalies.iterrows():
            date = anomaly["Date"]
            try:
                matching_indexes = signals.index[signals["Date"] == date]
                if matching_indexes.empty:
                    continue

                data_index = matching_indexes[0]
                if data_index < self.window_size:
                    continue

                history = signals.loc[data_index - self.window_size : data_index - 1, "Close"]
                past_trend = history.pct_change().mean() if len(history) > 1 else 0.0
                current_change = self._current_change(signals, anomaly, data_index)
                anomaly_score = self._anomaly_score(anomaly, current_change)

                trend_threshold = 1.5
                if current_change > past_trend * trend_threshold and current_change > 0:
                    signals.loc[data_index, "signal"] = 1
                elif current_change < past_trend * trend_threshold and current_change < 0:
                    signals.loc[data_index, "signal"] = -1
                signals.loc[data_index, "anomaly_score"] = np.float64(anomaly_score)
            except (IndexError, KeyError, TypeError, ValueError):
                logger.exception("シグナル生成に失敗しました（日付: %s）", date)

        return signals

    @staticmethod
    def _current_change(signals: pd.DataFrame, anomaly: pd.Series, data_index: int) -> float:
        if "pct_change" in anomaly:
            return float(anomaly["pct_change"]) / 100
        if data_index == 0:
            return 0.0

        previous_price = float(signals.loc[data_index - 1, "Close"])
        current_price = float(anomaly["Close"])
        return (current_price - previous_price) / previous_price

    @staticmethod
    def _anomaly_score(anomaly: pd.Series, current_change: float) -> float:
        if "z_score" in anomaly:
            return abs(float(anomaly["z_score"]))
        if "anomaly_score" in anomaly:
            return abs(float(anomaly["anomaly_score"]))
        return abs(current_change) / 0.01

    def forecast_adjustment(
        self,
        signals: pd.DataFrame,
        forecast: pd.DataFrame | None,
        weight: float = 0.5,
    ) -> pd.DataFrame:
        """シグナルの方向と強度に応じて将来予測を補正する。"""
        if forecast is None or forecast.empty:
            return pd.DataFrame()

        adjusted = forecast.copy()
        adjusted["predicted_price"] = adjusted["predicted_price"].astype(np.float64)

        for _, row in signals.loc[signals["signal"] != 0].iterrows():
            try:
                future_indexes = adjusted.index[adjusted["Date"] >= row["Date"]]
                for offset, future_index in enumerate(future_indexes):
                    decay = np.exp(-0.1 * offset)
                    adjustment = int(row["signal"]) * float(row["anomaly_score"]) * weight * decay
                    current_prediction = float(adjusted.loc[future_index, "predicted_price"])
                    adjusted.loc[future_index, "predicted_price"] = np.float64(
                        current_prediction * (1 + adjustment / 100)
                    )
            except (KeyError, TypeError, ValueError):
                logger.exception("予測補正に失敗しました")

        return adjusted
