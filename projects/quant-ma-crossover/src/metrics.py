import numpy as np
import pandas as pd


def calculate_metrics(data: pd.DataFrame) -> dict:
    returns = data["strategy_return"]
    equity = data["strategy_equity"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / len(data)) - 1
    annual_volatility = returns.std() * np.sqrt(252)

    if annual_volatility == 0 or np.isnan(annual_volatility):
        sharpe_ratio = 0
    else:
        sharpe_ratio = annual_return / annual_volatility

    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()
    number_of_trades = int(data["trade"].sum())

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
        "Number of Trades": number_of_trades,
    }
