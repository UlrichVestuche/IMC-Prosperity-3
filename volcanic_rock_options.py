import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# This function calculates option prices using the Black-Scholes model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price using Black-Scholes formula
    
    Parameters:
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility
    option_type: 'call' or 'put'
    
    Returns:
    Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Assuming we have volcanic_rock DataFrame with 'mid_price' and 'volatility' columns
# For this example, we'll use the most recent values
current_price = volcanic_rock['mid_price'].iloc[-1]
current_volatility = volcanic_rock['volatility'].iloc[-1]

# If volatility is NaN (which can happen with rolling calculations), use the average
if np.isnan(current_volatility):
    current_volatility = average_volatility

# Parameters for option pricing
risk_free_rate = 0.01  # 1% annual risk-free rate (assumption)
days_to_maturity = 7
maturity_in_years = days_to_maturity / 365  # Convert days to years

# Create a range of strike prices around the current price (±10%)
strike_prices = np.linspace(current_price * 0.9, current_price * 1.1, 100)

# Calculate call and put option prices for each strike price
call_prices = [black_scholes(current_price, K, maturity_in_years, risk_free_rate, current_volatility, 'call') for K in strike_prices]
put_prices = [black_scholes(current_price, K, maturity_in_years, risk_free_rate, current_volatility, 'put') for K in strike_prices]

# Plot the option prices
plt.figure(figsize=(12, 6))
plt.plot(strike_prices, call_prices, 'b-', label='Call Option')
plt.plot(strike_prices, put_prices, 'r-', label='Put Option')
plt.axvline(x=current_price, color='g', linestyle='--', label=f'Current Price: {current_price:.2f}')
plt.title(f'VOLCANIC_ROCK Option Prices (7-day maturity, volatility: {current_volatility:.4f})')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Time-series of option prices
# We'll calculate option prices using at-the-money options for each time point
# Convert time ticks to days for plotting
days_index = volcanic_rock.index / 10000  # each day is 10000 ticks

# Filter out NaN volatilities
valid_data = volcanic_rock.dropna(subset=['volatility'])

# Calculate at-the-money call and put option prices throughout the time series
atm_call_prices = []
atm_put_prices = []

for i in range(len(valid_data)):
    price = valid_data['mid_price'].iloc[i]
    vol = valid_data['volatility'].iloc[i]
    
    # Calculate ATM option prices (strike = current price)
    call_price = black_scholes(price, price, maturity_in_years, risk_free_rate, vol, 'call')
    put_price = black_scholes(price, price, maturity_in_years, risk_free_rate, vol, 'put')
    
    atm_call_prices.append(call_price)
    atm_put_prices.append(put_price)

# Plot the time series of option prices
plt.figure(figsize=(12, 6))
plt.plot(valid_data.index / 10000, atm_call_prices, 'b-', label='ATM Call Option')
plt.plot(valid_data.index / 10000, atm_put_prices, 'r-', label='ATM Put Option')
plt.title('At-the-Money Option Prices Over Time (7-day maturity)')
plt.xlabel('Days')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Volatility surface
# For this, we'll use different strikes and different expiry times
expiry_days = [1, 3, 5, 7, 10, 14]
expiry_years = [d/365 for d in expiry_days]
strike_range = np.linspace(0.8, 1.2, 5)  # 80% to 120% of current price

# Create the volatility surface plot
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='3d')

for i, T in enumerate(expiry_years):
    for j, k_ratio in enumerate(strike_range):
        K = current_price * k_ratio
        call = black_scholes(current_price, K, T, risk_free_rate, current_volatility, 'call')
        ax.scatter(K, expiry_days[i], call, c='b', marker='o')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Days to Expiry')
ax.set_zlabel('Call Option Price')
ax.set_title('Call Option Price Surface')
plt.tight_layout()
plt.show() 