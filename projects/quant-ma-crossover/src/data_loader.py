import yfinance as yf
import pandas as pd


def download_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Close"]].copy()
    data.rename(columns={"Close": "close"}, inplace=True)
    data.dropna(inplace=True)

    return data
