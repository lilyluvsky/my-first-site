# Quant Moving Average Crossover Backtest

This project implements a simple quantitative trading backtest using Python.

## Strategy

The strategy uses a moving average crossover:

- Buy when the short moving average is above the long moving average.
- Stay in cash when the short moving average is below the long moving average.

## Features

- Downloads historical market data using yfinance
- Generates trading signals
- Runs a vectorised backtest
- Includes transaction costs
- Calculates performance metrics
- Saves results to CSV
- Plots an equity curve

## Example Result

Backtest asset: SPY  
Strategy: 20/50 moving average crossover

Metrics include:

- Total return
- Annual return
- Annual volatility
- Sharpe ratio
- Maximum drawdown
- Win rate
- Number of trades

## Run

```bash
pip install -r requirements.txt
python main.py
```
