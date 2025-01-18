import yfinance as yf
import pandas as pd

# Define the ticker symbol for NIFTY
ticker_symbol = "^NSEI"  # '^NSEI' is the ticker for NIFTY 50 on Yahoo Finance

# Fetch historical data for the past 3 years
nifty_data = yf.download(ticker_symbol, start="2022-01-01", end="2025-01-01", interval="1d")

# Save the data to a CSV file
nifty_data.to_csv("NIFTY_historical_data.csv")

# Display the first few rows
print(nifty_data.head())
