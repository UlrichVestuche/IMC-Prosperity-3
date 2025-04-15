import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes option pricing model
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
    # Handle edge case with very low volatility
    if sigma <= 0:
        if option_type == 'call':
            return max(0, S - K * np.exp(-r * T))
        else:
            return max(0, K * np.exp(-r * T) - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Assuming we have volcanic_rock DataFrame with the required columns
# Convert the index to days (each day is 10000 ticks)
volcanic_rock['day'] = volcanic_rock.index / 10000

# Compute time to maturity in years for each data point
# 7 days maturity from each point in time
volcanic_rock['time_to_maturity'] = 7 / 365  # 7 days converted to years

# Parameters for option pricing
risk_free_rate = 0.01  # 1% annual risk-free rate (assumption)

# Calculate option prices at each data point
call_prices = []
put_prices = []

for i in range(len(volcanic_rock)):
    row = volcanic_rock.iloc[i]
    S = row['mid_price']
    sigma = row['volatility']
    T = row['time_to_maturity']
    
    # Skip if any required value is NaN
    if np.isnan(S) or np.isnan(sigma) or np.isnan(T):
        call_prices.append(np.nan)
        put_prices.append(np.nan)
        continue
    
    # Handle very low volatility values
    if sigma <= 0:
        sigma = 0.0001  # Set a minimum volatility
        
    # Calculate ATM option prices (strike = current price)
    call_price = black_scholes(S, S, T, risk_free_rate, sigma, 'call')
    put_price = black_scholes(S, S, T, risk_free_rate, sigma, 'put')
    
    call_prices.append(call_price)
    put_prices.append(put_price)

# Add option prices to the DataFrame
volcanic_rock['call_option_price'] = call_prices
volcanic_rock['put_option_price'] = put_prices

# Plot the option prices over time
plt.figure(figsize=(14, 8))

# Plot call option prices
plt.subplot(2, 1, 1)
plt.plot(volcanic_rock['day'], volcanic_rock['call_option_price'], 'b-', label='Call Option Price')
plt.title('VOLCANIC_ROCK Call Option Prices (7-day maturity)')
plt.xlabel('Days')
plt.ylabel('Option Price')
plt.grid(True)
plt.legend()

# Plot put option prices
plt.subplot(2, 1, 2)
plt.plot(volcanic_rock['day'], volcanic_rock['put_option_price'], 'r-', label='Put Option Price')
plt.title('VOLCANIC_ROCK Put Option Prices (7-day maturity)')
plt.xlabel('Days')
plt.ylabel('Option Price')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Calculate and plot option prices for different strike prices at the latest data point
latest_data = volcanic_rock.iloc[-1]
S = latest_data['mid_price']
sigma = latest_data['volatility']
T = latest_data['time_to_maturity']

# If volatility is NaN, use the average
if np.isnan(sigma):
    sigma = average_volatility

# Create a range of strike prices around the current price (±20%)
strike_prices = np.linspace(S * 0.8, S * 1.2, 100)

# Calculate call and put option prices for each strike price
strike_call_prices = [black_scholes(S, K, T, risk_free_rate, sigma, 'call') for K in strike_prices]
strike_put_prices = [black_scholes(S, K, T, risk_free_rate, sigma, 'put') for K in strike_prices]

# Plot option prices vs strike prices
plt.figure(figsize=(12, 6))
plt.plot(strike_prices, strike_call_prices, 'b-', label='Call Option')
plt.plot(strike_prices, strike_put_prices, 'r-', label='Put Option')
plt.axvline(x=S, color='g', linestyle='--', label=f'Current Price: {S:.2f}')
plt.title(f'VOLCANIC_ROCK Option Prices vs Strike (Volatility: {sigma:.4f}, 7-day maturity)')
plt.xlabel('Strike Price')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot option prices vs volatility (sensitivity analysis)
# Use the latest price but vary the volatility
volatilities = np.linspace(0.001, 0.3, 100)  # Range of volatilities

# Calculate ATM option prices for each volatility
vol_call_prices = [black_scholes(S, S, T, risk_free_rate, vol, 'call') for vol in volatilities]
vol_put_prices = [black_scholes(S, S, T, risk_free_rate, vol, 'put') for vol in volatilities]

# Plot option prices vs volatility
plt.figure(figsize=(12, 6))
plt.plot(volatilities, vol_call_prices, 'b-', label='Call Option')
plt.plot(volatilities, vol_put_prices, 'r-', label='Put Option')
plt.axvline(x=sigma, color='g', linestyle='--', label=f'Current Volatility: {sigma:.4f}')
plt.title(f'Option Prices vs Volatility (Price: {S:.2f}, 7-day maturity)')
plt.xlabel('Volatility')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 