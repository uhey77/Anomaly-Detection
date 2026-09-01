"""分析で使用する市場データの読み込みと正規化。"""

from pathlib import Path

import pandas as pd

from utils.data_utils import DataManager

_DEFAULT_DATA_MANAGER = DataManager()


def prepare_data(
    use_sample: bool,
    file_path: str | Path | None = None,
    symbol: str = "sp500",
    include_extra_indicators: bool = True,
    data_manager: DataManager | None = None,
) -> pd.DataFrame:
    """サンプルまたはアップロードされた市場データを分析可能な形にする。"""
    manager = data_manager or _DEFAULT_DATA_MANAGER
    if use_sample:
        frame = _load_market_data(manager, symbol, include_extra_indicators)
    else:
        frame = _load_uploaded_data(file_path)

    if frame.empty:
        raise ValueError("有効なデータが取得できませんでした")
    if "Date" not in frame or "Close" not in frame:
        raise ValueError("Date列とClose列が必要です")

    normalized = frame.copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["Close"] = pd.to_numeric(normalized["Close"], errors="coerce")
    normalized = normalized.dropna(subset=["Date", "Close"])
    if normalized.empty:
        raise ValueError("有効な日付・価格データがありません")

    return normalized.sort_values("Date").reset_index(drop=True)


def _load_market_data(
    manager: DataManager,
    symbol: str,
    include_extra_indicators: bool,
) -> pd.DataFrame:
    if not include_extra_indicators:
        return manager.load_sample_data(symbol=symbol)

    data_by_symbol = manager.load_multi_indicator_data()
    main_symbol = symbol if symbol in data_by_symbol else "sp500"
    if main_symbol not in data_by_symbol:
        raise ValueError(f"市場データが見つかりません: {symbol}")

    frame = data_by_symbol[main_symbol].copy()
    indicator_columns = {
        "volume": "Volume",
        "vix": "VIX",
        "usdjpy": "USDJPY",
    }
    for indicator, column in indicator_columns.items():
        indicator_frame = data_by_symbol.get(indicator)
        if (
            column not in frame
            and indicator_frame is not None
            and not indicator_frame.empty
            and column in indicator_frame
        ):
            frame = frame.merge(
                indicator_frame[["Date", column]],
                on="Date",
                how="left",
            )
    return frame


def _load_uploaded_data(file_path: str | Path | None) -> pd.DataFrame:
    if not file_path:
        raise ValueError("ファイルがアップロードされていません")

    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".xlsx":
        frame = pd.read_excel(path)
    else:
        raise ValueError("対応していないファイル形式です。CSVまたはXLSXを使用してください")

    if len(frame.columns) < 2:
        raise ValueError("日付列と価格列を含むファイルが必要です")
    if "Date" not in frame:
        frame = frame.rename(columns={frame.columns[0]: "Date"})
    if "Close" not in frame:
        value_columns = [column for column in frame.columns if column != "Date"]
        frame = frame.rename(columns={value_columns[0]: "Close"})
    return frame
