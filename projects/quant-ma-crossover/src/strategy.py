import pandas as pd


def moving_average_crossover(
    data: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50
) -> pd.DataFrame:
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    df = data.copy()

    df["short_ma"] = df["close"].rolling(short_window).mean()
    df["long_ma"] = df["close"].rolling(long_window).mean()

    df["signal"] = 0
    df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1

    # Shift position by one day to avoid look-ahead bias
    df["position"] = df["signal"].shift(1).fillna(0)

    return df
