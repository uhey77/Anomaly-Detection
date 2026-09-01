import gc
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import app_improved
from services.data_service import prepare_data
from utils.llm_clients import HuggingFaceClient, OpenAIClient
from utils.signal_generator import SignalGenerator


class CoreBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random = np.random.default_rng(42)
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        values = 100 + np.linspace(0, 5, 100) + random.normal(0, 0.5, 100)
        values[50] += 15
        cls.data = pd.DataFrame(
            {
                "Date": dates,
                "Close": values,
                "Volume": np.arange(100) + 1000,
            }
        )

    def test_detection_results_remain_stable(self):
        expected_indexes = {
            ("z_score", 2.0): [50, 51],
            ("iqr", 1.5): [50, 51],
            ("moving_avg", 2.0): [30, 50, 51, 98],
            ("isolation_forest", 3.0): [4, 22, 30, 31, 50, 51, 52, 57, 58, 60],
        }

        for (method, threshold), expected in expected_indexes.items():
            with self.subTest(method=method):
                anomalies = app_improved.detect_anomalies(
                    self.data,
                    method,
                    threshold,
                    extra_indicators=False,
                )
                self.assertEqual(list(anomalies.index), expected)

    def test_signal_generation_remains_stable(self):
        anomalies = app_improved.detect_anomalies(
            self.data,
            "z_score",
            2.0,
            extra_indicators=False,
        )
        signals = SignalGenerator(window_size=5).generate_signals(
            self.data,
            anomalies,
        )
        self.assertEqual(list(signals.index[signals["signal"] != 0]), [50, 51])

    def test_uploaded_data_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            pd.DataFrame(
                {
                    "timestamp": ["2025-01-02", "2025-01-01"],
                    "price": ["101.5", "100.0"],
                }
            ).to_csv(path, index=False)

            loaded = prepare_data(use_sample=False, file_path=path)

        self.assertEqual(list(loaded.columns), ["Date", "Close"])
        self.assertEqual(list(loaded["Close"]), [100.0, 101.5])
        self.assertTrue(loaded["Date"].is_monotonic_increasing)

    def test_xlsx_data_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.xlsx"
            pd.DataFrame(
                {
                    "Date": ["2025-01-01", "2025-01-02"],
                    "Close": [100.0, 101.5],
                }
            ).to_excel(path, index=False)

            loaded = prepare_data(use_sample=False, file_path=path)

        self.assertEqual(list(loaded["Close"]), [100.0, 101.5])

    def test_analysis_passes_extra_indicator_setting_by_name(self):
        empty_anomalies = pd.DataFrame()
        progress = Mock()

        with (
            patch("app_improved.prepare_data", return_value=self.data) as prepare,
            patch("app_improved.detect_anomalies", return_value=empty_anomalies),
        ):
            result = app_improved.run_analysis(
                "sample",
                None,
                "z_score",
                2.0,
                "mock",
                False,
                False,
                False,
                False,
                False,
                include_extra_indicators=False,
                generate_signals=False,
                progress=progress,
            )

        prepare.assert_called_once_with(
            True,
            None,
            include_extra_indicators=False,
        )
        self.assertIsInstance(result[0], go.Figure)

    def test_ui_can_be_constructed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            interface = app_improved.create_gradio_ui()
            self.assertIsInstance(interface, gr.Blocks)
            interface.close()
            del interface
            gc.collect()

    def test_llm_clients_honor_explicit_api_keys(self):
        openai = OpenAIClient(api_key="openai-test", model="test-model")
        hugging_face = HuggingFaceClient(api_key="hf-test", model="test-model")
        self.assertEqual(openai.api_key, "openai-test")
        self.assertEqual(hugging_face.api_key, "hf-test")


if __name__ == "__main__":
    unittest.main()
