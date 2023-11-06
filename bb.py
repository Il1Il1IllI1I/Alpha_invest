# 1. 필요한 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2. 데이터 로드하기
close_df = pd.read_csv('reshape_df.csv')
max_date = close_df['Date'].max()

# 3. 필요한 설정 값을 초기화하기
rebalancing_dates = ['2013-04-01', '2014-04-01', '2015-04-01', '2016-03-31', '2017-04-03', '2018-04-02', '2019-04-01', 
                     '2020-04-01', '2021-04-01', '2022-04-01', '2023-04-03']
initial_capital = 100000000  # 1억 원
transaction_fee = 0.0015  # 0.15%
N = 50  # 이동평균 기간
M = 10  # 포트폴리오에 포함할 종목 수

# 4. 전략 함수 정의하기
def get_selected_stocks_for_mr(period_df):
    price_drop = (period_df.iloc[-1] - period_df.iloc[0]) / period_df.iloc[0]
    dropped_stocks = price_drop[price_drop <= -0.30].index.tolist()
    volatility = period_df[dropped_stocks].iloc[-60:].std()
    return volatility.nsmallest(10).index.tolist() if len(volatility) >= 10 else volatility.index.tolist()

def get_selected_stocks_for_tf(period_df):
    moving_average = period_df.rolling(window=N).mean().iloc[-1]
    current_price = period_df.iloc[-1]
    buy_signals = current_price[current_price > moving_average].index.tolist()
    volatility = period_df[buy_signals].iloc[-60:].std()
    return volatility.nsmallest(M).index.tolist() if len(volatility) >= M else volatility.index.tolist()

# 전략 실행 함수
def portfolio_strategy_execution(strategy_function, rebalancing_dates, close_df):
    portfolio_values = [initial_capital]
    held_stocks_list = []
    
    for idx, date in enumerate(rebalancing_dates):
        start_date = pd.to_datetime(date) - pd.DateOffset(years=1)
        end_date = pd.to_datetime(date)
        period_df = close_df[(close_df['Date'] >= start_date.strftime('%Y-%m-%d')) & (close_df['Date'] < end_date.strftime('%Y-%m-%d'))].drop(columns='Date')
        selected_stocks = strategy_function(period_df)
        
        held_stocks_list.append(selected_stocks)
        amount_per_stock = (portfolio_values[-1] * (1 - transaction_fee)) / len(selected_stocks)
        next_date = rebalancing_dates[idx + 1] if idx + 1 < len(rebalancing_dates) else max_date
        next_period_df = close_df[(close_df['Date'] >= date) & (close_df['Date'] <= next_date)].drop(columns='Date')
        returns = (next_period_df[selected_stocks].iloc[-1] - next_period_df[selected_stocks].iloc[0]) / next_period_df[selected_stocks].iloc[0]
        portfolio_value = sum(amount_per_stock * (1 + returns) * (1 - transaction_fee))
        portfolio_values.append(portfolio_value)
    
    return portfolio_values, held_stocks_list

# 5. 성과 평가 지표 함수 정의하기
def performance_metrics(portfolio_values, rebalancing_dates):
    portfolio_values = [value for value in portfolio_values if value > 0]  # 0인 값 제거
    daily_returns = [0] + [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] for i in range(1, len(portfolio_values))]
    cagr = ((portfolio_values[-1] / portfolio_values[0]) ** (1/len(rebalancing_dates))) - 1
    rolling_max = np.maximum.accumulate(portfolio_values)
    daily_drawdown = portfolio_values / rolling_max - 1.0
    mdd = np.min(daily_drawdown)
    sharpe_ratio = np.mean(daily_returns[1:]) / np.std(daily_returns[1:]) * np.sqrt(len(rebalancing_dates))
    return cagr, mdd, sharpe_ratio

# 6. 전략을 실행하고 성과 지표를 계산하기
mr_portfolio_values, mr_held_stocks = portfolio_strategy_execution(get_selected_stocks_for_mr, rebalancing_dates, close_df)
tf_portfolio_values, tf_held_stocks = portfolio_strategy_execution(get_selected_stocks_for_tf, rebalancing_dates, close_df)
mr_metrics = performance_metrics(mr_portfolio_values, rebalancing_dates)
tf_metrics = performance_metrics(tf_portfolio_values, rebalancing_dates)

# 7. 결과를 시각화하기
plt.figure(figsize=(18, 12))
plt.plot(rebalancing_dates + [max_date], mr_portfolio_values, marker='o', linestyle='-', color='b', label='Mean Reversion Portfolio Value')
for i, date in enumerate(rebalancing_dates):
    stocks_str = ', '.join(mr_held_stocks[i])
    plt.annotate(stocks_str, (date, mr_portfolio_values[i+1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='darkblue')

plt.plot(rebalancing_dates + [max_date], tf_portfolio_values, marker='o', linestyle='-', color='g', label='Trend Following Portfolio Value')
for i, date in enumerate(rebalancing_dates):
    stocks_str = ', '.join(tf_held_stocks[i])
    plt.annotate(stocks_str, (date, tf_portfolio_values[i+1]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='darkgreen')

plt.title('Portfolio Value and Stocks Held at Each Rebalancing Date', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Portfolio Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--", c='0.65')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
