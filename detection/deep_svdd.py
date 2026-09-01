"""オートエンコーダーを使ったDeep SVDD風の時系列異常検知。"""

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DeepSVDDDetector:
    """再構成誤差の上位百分位を異常として検出する。"""

    def __init__(
        self,
        threshold: float = 0.9,
        epochs: int = 20,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.threshold = threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model: tf.keras.Model | None = None

    def detect(
        self,
        data: pd.DataFrame,
        date_column: str = "Date",
        value_column: str = "Close",
        extra_features: list[str] | None = None,
    ) -> pd.DataFrame:
        """入力データから異常と判定された行だけを返す。"""
        del date_column  # API互換のため引数を維持する

        if data.empty or len(data) < 20:
            logger.warning("Deep SVDD用データが不足しています（%d件）", len(data))
            return pd.DataFrame()
        if value_column not in data:
            raise ValueError(f"必須カラムがありません: {value_column}")

        try:
            frame = data.copy()
            frame["returns"] = frame[value_column].pct_change().fillna(0)
            frame["log_returns"] = np.log1p(frame["returns"])

            rolling_window = min(5, len(frame) // 4)
            frame["rolling_mean"] = (
                frame[value_column].rolling(window=rolling_window, min_periods=1).mean()
            )
            frame["rolling_std"] = (
                frame[value_column].rolling(window=rolling_window, min_periods=1).std()
            )
            frame["rolling_z"] = (frame[value_column] - frame["rolling_mean"]) / (
                frame["rolling_std"] + 1e-8
            )

            average_window = min(10, len(frame) // 2)
            frame["ma"] = frame[value_column].rolling(window=average_window, min_periods=1).mean()
            frame["ma_ratio"] = frame[value_column] / (frame["ma"] + 1e-8)

            features = ["returns", "log_returns", "rolling_z", "ma_ratio"]
            features.extend(
                feature for feature in (extra_features or []) if feature in frame.columns
            )

            feature_frame = frame.dropna().copy()
            if len(feature_frame) < 10:
                logger.warning(
                    "Deep SVDD特徴量処理後のデータが不足しています（%d件）",
                    len(feature_frame),
                )
                return pd.DataFrame()

            values = feature_frame[features].to_numpy()
            if not np.isfinite(values).all():
                values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
            scaled_values = StandardScaler().fit_transform(values)

            tf.random.set_seed(self.random_state)
            input_dimension = scaled_values.shape[1]
            encoder = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(input_dimension,)),
                    tf.keras.layers.Dense(max(2, input_dimension // 2), activation="relu"),
                    tf.keras.layers.Dense(2, activation="linear"),
                ]
            )
            decoder = tf.keras.Sequential(
                [
                    tf.keras.Input(shape=(2,)),
                    tf.keras.layers.Dense(max(2, input_dimension // 2), activation="relu"),
                    tf.keras.layers.Dense(input_dimension, activation="linear"),
                ]
            )
            self.model = tf.keras.Sequential([encoder, decoder])
            self.model.compile(optimizer="adam", loss="mse")

            batch_size = max(1, min(self.batch_size, len(scaled_values) // 2))
            self.model.fit(
                scaled_values,
                scaled_values,
                epochs=self.epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.1 if len(scaled_values) > 10 else 0,
            )

            reconstructed = self.model.predict(scaled_values, verbose=0)
            errors = np.mean(np.square(scaled_values - reconstructed), axis=1)
            error_threshold = np.percentile(errors, self.threshold * 100)

            feature_frame["anomaly_score"] = errors
            feature_frame["is_anomaly"] = errors > error_threshold
            feature_frame["pct_change"] = feature_frame[value_column].pct_change() * 100
            return feature_frame.loc[feature_frame["is_anomaly"]].copy()
        except (TypeError, ValueError, KeyError):
            logger.exception("Deep SVDD検出中にエラーが発生しました")
            return pd.DataFrame()
