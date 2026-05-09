import matplotlib.pyplot as plt

from src.data_loader import download_data
from src.strategy import moving_average_crossover
from src.backtest import run_backtest
from src.metrics import calculate_metrics


def main():
    symbol = "SPY"
    start = "2015-01-01"
    end = "2025-01-01"

    short_window = 20
    long_window = 50

    data = download_data(symbol, start, end)

    strategy_data = moving_average_crossover(
        data,
        short_window=short_window,
        long_window=long_window
    )

    results = run_backtest(
        strategy_data,
        initial_capital=10000,
        transaction_cost=0.001
    )

    metrics = calculate_metrics(results)

    print()
    print(f"Backtest Results: {symbol}")
    print(f"Strategy: MA Crossover ({short_window}/{long_window})")
    print("-" * 50)

    for key, value in metrics.items():
        if key == "Sharpe Ratio":
            print(f"{key}: {value:.2f}")
        elif key == "Number of Trades":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2%}")

    results.to_csv("results/backtest_results.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(results.index, results["market_equity"], label="Buy and Hold")
    plt.plot(results.index, results["strategy_equity"], label="MA Strategy")
    plt.title(f"{symbol} Moving Average Crossover Backtest")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/equity_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
