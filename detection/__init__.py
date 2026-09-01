"""異常検知器の公開インターフェース。"""

from collections.abc import Mapping
from typing import Any

from .anomaly_detector import AnomalyDetector
from .deep_svdd import DeepSVDDDetector
from .isolation_forest import IsolationForestDetector

Detector = AnomalyDetector | DeepSVDDDetector | IsolationForestDetector


def create_detector(
    method: str,
    threshold: float = 3.0,
    params: Mapping[str, Any] | None = None,
) -> Detector:
    """検出方法に応じた検出器を、許可された設定だけで生成する。"""
    settings = dict(params or {})

    if method == "isolation_forest":
        return IsolationForestDetector(
            contamination=settings.get("contamination", 0.05),
            n_estimators=settings.get("n_estimators", 100),
            random_state=settings.get("random_state", 42),
        )
    if method == "deep_svdd":
        return DeepSVDDDetector(
            threshold=settings.get("threshold", 0.9),
            epochs=settings.get("epochs", 20),
            batch_size=settings.get("batch_size", 32),
            random_state=settings.get("random_state", 42),
        )

    return AnomalyDetector(method=method, threshold=float(threshold), **settings)


__all__ = [
    "AnomalyDetector",
    "DeepSVDDDetector",
    "IsolationForestDetector",
    "create_detector",
]
