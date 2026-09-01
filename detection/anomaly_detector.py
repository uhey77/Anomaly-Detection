"""統計手法と拡張モデルを統一的に扱う異常検知器。"""

import pandas as pd
from scipy import stats

SUPPORTED_METHODS = {
    "z_score",
    "iqr",
    "moving_avg",
    "isolation_forest",
    "deep_svdd",
}


class AnomalyDetector:
    """金融時系列データに複数の異常検知手法を適用する。"""

    def __init__(
        self,
        method: str = "z_score",
        threshold: float = 3.0,
        **kwargs,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            supported = ", ".join(sorted(SUPPORTED_METHODS))
            raise ValueError(f"未対応の検出方法です: {method}（対応: {supported}）")

        self.method = method
        self.threshold = threshold
        self.kwargs = kwargs
        self.detector = self._create_specialized_detector()

    def _create_specialized_detector(self):
        if self.method == "isolation_forest":
            from .isolation_forest import IsolationForestDetector

            return IsolationForestDetector(
                contamination=self.kwargs.get("contamination", 0.05),
                n_estimators=self.kwargs.get("n_estimators", 100),
                random_state=self.kwargs.get("random_state", 42),
            )
        if self.method == "deep_svdd":
            from .deep_svdd import DeepSVDDDetector

            return DeepSVDDDetector(
                threshold=self.kwargs.get("threshold", 0.9),
                epochs=self.kwargs.get("epochs", 20),
                batch_size=self.kwargs.get("batch_size", 32),
                random_state=self.kwargs.get("random_state", 42),
            )
        return None

    def detect(
        self,
        data: pd.DataFrame,
        date_column: str = "Date",
        value_column: str = "Close",
        extra_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """入力データから異常と判定された行だけを返す。"""
        if self.detector is not None:
            return self.detector.detect(
                data,
                date_column,
                value_column,
                extra_features,
            )
        if value_column not in data:
            raise ValueError(f"必須カラムがありません: {value_column}")

        frame = data.copy()
        values = frame[value_column]

        if self.method == "z_score":
            frame["z_score"] = stats.zscore(values)
            frame["is_anomaly"] = frame["z_score"].abs() > self.threshold
        elif self.method == "iqr":
            first_quartile = values.quantile(0.25)
            third_quartile = values.quantile(0.75)
            interquartile_range = third_quartile - first_quartile
            lower_bound = first_quartile - self.threshold * interquartile_range
            upper_bound = third_quartile + self.threshold * interquartile_range
            frame["is_anomaly"] = values.lt(lower_bound) | values.gt(upper_bound)
        else:
            window = self.kwargs.get("window", 20)
            frame["moving_avg"] = values.rolling(window=window).mean()
            frame["moving_std"] = values.rolling(window=window).std()
            frame["is_anomaly"] = (
                (values - frame["moving_avg"]).abs() > self.threshold * frame["moving_std"]
            ).fillna(False)

        frame["pct_change"] = values.pct_change() * 100
        change_threshold = self.threshold * frame["pct_change"].std()
        frame["is_significant_change"] = (frame["pct_change"].abs() > change_threshold).fillna(
            False
        )
        frame["is_anomaly"] |= frame["is_significant_change"]
        return frame.loc[frame["is_anomaly"]].copy()
