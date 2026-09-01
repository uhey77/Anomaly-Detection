"""Isolation Forestを使った時系列異常検知。"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """価格変化と移動平均からIsolation Forestで異常を検出する。"""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: IsolationForest | None = None

    def detect(
        self,
        data: pd.DataFrame,
        date_column: str = "Date",
        value_column: str = "Close",
        extra_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """入力データから異常と判定された行だけを返す。"""
        del date_column  # API互換のため引数を維持する

        if data.empty or len(data) < 5:
            logger.warning("Isolation Forest用データが不足しています（%d件）", len(data))
            return pd.DataFrame()
        if value_column not in data:
            raise ValueError(f"必須カラムがありません: {value_column}")

        try:
            frame = data.copy()
            frame["returns"] = frame[value_column].pct_change()
            frame["log_returns"] = np.log1p(frame["returns"].fillna(0))
            frame["rolling_mean_5"] = frame[value_column].rolling(window=5, min_periods=1).mean()
            frame["rolling_std_5"] = frame[value_column].rolling(window=5, min_periods=1).std()
            frame["rolling_z"] = (frame[value_column] - frame["rolling_mean_5"]) / (
                frame["rolling_std_5"] + 1e-8
            )

            for window in (20, 50):
                average_column = f"ma_{window}"
                ratio_column = f"ma_ratio_{window}"
                frame[average_column] = (
                    frame[value_column].rolling(window=window, min_periods=1).mean()
                )
                frame[ratio_column] = frame[value_column] / (frame[average_column] + 1e-8)

            features = [
                "returns",
                "log_returns",
                "rolling_z",
                "ma_ratio_20",
                "ma_ratio_50",
            ]
            features.extend(
                feature for feature in (extra_features or []) if feature in frame.columns
            )

            feature_frame = frame.dropna().copy()
            if len(feature_frame) < 3:
                logger.warning(
                    "Isolation Forest特徴量処理後のデータが不足しています（%d件）",
                    len(feature_frame),
                )
                return pd.DataFrame()

            values = feature_frame[features].to_numpy()
            if not np.isfinite(values).all():
                values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)

            scaled_values = StandardScaler().fit_transform(values)
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            self.model.fit(scaled_values)

            feature_frame["anomaly_score"] = self.model.score_samples(scaled_values)
            feature_frame["is_anomaly"] = self.model.predict(scaled_values) == -1
            feature_frame["pct_change"] = feature_frame[value_column].pct_change() * 100
            return feature_frame.loc[feature_frame["is_anomaly"]].copy()
        except (TypeError, ValueError, KeyError):
            logger.exception("Isolation Forest検出中にエラーが発生しました")
            return pd.DataFrame()
