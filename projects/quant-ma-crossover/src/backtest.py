import pandas as pd


def run_backtest(
    data: pd.DataFrame,
    initial_capital: float = 10000,
    transaction_cost: float = 0.001
) -> pd.DataFrame:
    df = data.copy()

    df["market_return"] = df["close"].pct_change()

    df["trade"] = df["position"].diff().abs().fillna(0)
    df["strategy_return_before_cost"] = df["position"] * df["market_return"]
    df["cost"] = df["trade"] * transaction_cost

    df["strategy_return"] = df["strategy_return_before_cost"] - df["cost"]

    df["market_equity"] = initial_capital * (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()

    df.dropna(inplace=True)

    return df
