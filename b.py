import yfinance as yf
import pandas as pd

# Download data for a single ticker (e.g., MSFT)
# Specify the ticker symbol, start date, and end date
data = yf.download('MSFT', start='2019-01-01', end='2024-01-01')

# View the first few rows of the downloaded data
print(data.head())

# Optionally, save the data to a CSV file
data.to_csv('msft_stock_data.csv')
