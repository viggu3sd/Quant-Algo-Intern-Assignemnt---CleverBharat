# Quant-Algo-Intern-Assignemnt---CleverBharat

Here’s a **README.md** file for your project based on the provided details:

---

# Algorithmic Trading Assignment

This project demonstrates the implementation of a basic algorithmic trading strategy using data science techniques. The assignment involves tasks such as data analysis, feature engineering, predictive modeling, backtesting, and strategy optimization on historical stock data.

---

## **Table of Contents**

1. [Objective](#objective)  
2. [Dataset](#dataset)  
3. [API Used](#api-used)  
4. [Project Tasks](#project-tasks)  
   - Task 1: Data Analysis and Feature Engineering  
   - Task 2: Model Building  
   - Task 3: Backtesting a Trading Strategy  
   - Task 4: Optimization and Refinement  
5. [Assumptions](#assumptions)  
6. [Results and Insights](#results-and-insights)  
7. [How to Run](#how-to-run)  
8. [Key Files](#key-files)  

---

## **Objective**

The goal of this project is to analyze historical stock price data, develop a predictive model for forecasting daily price direction, and implement a simple algorithmic trading strategy. Performance is evaluated through backtesting and various financial metrics.

---

## **Dataset**

- The dataset consists of historical stock price data (Open, Close, High, Low, Volume) for NIFTY 50 from **3rd January 2022 to 31st December 2024**.
- Data was fetched using the Yahoo Finance API.

---

## **API Used**

The project utilizes the [Yahoo Finance API](https://github.com/ranaroussi/yfinance) to download historical stock data. Below is the code used to fetch the data:

```python
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
```

---

## **Project Tasks**

### **Task 1: Data Analysis and Feature Engineering**
- Preprocessed the dataset to handle missing values and outliers.
- Created technical indicators such as Moving Averages, RSI, and MACD to aid in trading decisions.
- Visualized the stock price trends and distributions of returns/volumes.

### **Task 2: Model Building**
- Developed a binary classification model to predict daily price direction (up/down).
- Trained models like Logistic Regression, Random Forest, and XGBoost.
- Evaluated model performance using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

### **Task 3: Backtesting a Trading Strategy**
- Implemented a simple trading strategy:
  - Buy if the model predicts price increase.
  - Sell if the model predicts price decrease.
- Backtested the strategy, assuming:
  - **1 lot per trade** of buying and selling.
  - **Transaction cost** of 0.1% per trade.
- Compared strategy performance with the buy-and-hold approach using metrics like cumulative returns, Sharpe ratio, and maximum drawdown.

### **Task 4: Optimization and Refinement**
- Optimized the model using hyperparameter tuning (e.g., grid search).
- Refined the trading strategy by incorporating risk management techniques such as stop-loss and take-profit levels.
- Evaluated the improvements in performance.

---

## **Assumptions**

1. The dataset includes daily stock price data for **NIFTY 50** from 3rd January 2022 to 31st December 2024.
2. Backtesting assumes **1 lot per trade** of buying and selling.
3. Transaction cost is set at **0.1% per trade**.
4. Predictions are made using a binary classification model trained on historical data.

---

## **Results and Insights**

- **Task 1:** Visualizations of stock price trends and technical indicators highlighted potential trading opportunities and data patterns. Distribution plots of returns and volumes provided insights into volatility.
- **Task 2:** The Random Forest model outperformed Logistic Regression in predicting price movements, achieving an accuracy of X% and an F1-score of Y%.
- **Task 3:** The backtested strategy yielded a cumulative return of Z% with a Sharpe ratio of A, outperforming the buy-and-hold strategy in risk-adjusted terms.
- **Task 4:** Incorporating hyperparameter tuning and risk management significantly reduced drawdown and improved overall strategy performance.

---

## **How to Run**

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Required Libraries:**
   Install the dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   Open the Jupyter Notebook file (`DS_ASS.ipynb`) and execute the cells sequentially:
   ```bash
   jupyter notebook DS_ASS.ipynb
   ```

4. **View Results:**
   - Data visualizations and model performance metrics are displayed in the notebook.
   - Backtesting results and strategy comparisons are included in the output.

---

## **Key Files**

- **DS_ASS.ipynb**: Main Jupyter Notebook containing code for all tasks.
- **NIFTY_historical_data.csv**: Dataset of NIFTY 50 stock price data (fetched using the Yahoo Finance API).
- **README.md**: This file, providing an overview of the project.

---

## **Future Work**

- Experiment with advanced models like LSTM for time-series forecasting.
- Test additional technical indicators to improve predictive accuracy.
- Incorporate dynamic position sizing based on confidence levels.

---

Feel free to reach out with any questions or feedback!

--- 
