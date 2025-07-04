import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2.) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def create_heatmap_data(base_S, base_sigma, K, T, r, price_range=0.3, vol_range=0.5, grid_size=20):
    """Create heatmap data by shocking stock price and volatility"""
    
    # Create ranges for stock price and volatility (configurable shocks)
    S_min = base_S * (1 - price_range)
    S_max = base_S * (1 + price_range)
    sigma_min = max(0.01, base_sigma * (1 - vol_range))
    sigma_max = base_sigma * (1 + vol_range)
    
    # Create grids
    S_range = np.linspace(S_min, S_max, grid_size)
    sigma_range = np.linspace(sigma_min, sigma_max, grid_size)
    
    # Initialize matrices for call and put prices
    call_prices = np.zeros((len(sigma_range), len(S_range)))
    put_prices = np.zeros((len(sigma_range), len(S_range)))
    
    # Calculate option prices for each combination
    for i, vol in enumerate(sigma_range):
        for j, price in enumerate(S_range):
            call_prices[i, j] = black_scholes_call(price, K, T, r, vol)
            put_prices[i, j] = black_scholes_put(price, K, T, r, vol)
    
    return S_range, sigma_range, call_prices, put_prices

# Streamlit App
st.title("Black-Scholes Option Pricer with Heatmaps")

# Sidebar for inputs
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Current Stock Price ($)", value=100.0, min_value=0.01)
K = st.sidebar.number_input("Strike Price ($)", value=100.0, min_value=0.01)
T = st.sidebar.number_input("Time to Expiry (Years)", value=1.0, min_value=0.01)
r = st.sidebar.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0)
sigma = st.sidebar.number_input("Volatility", value=0.20, min_value=0.01, max_value=5.0)

# Configurable shock parameters (as mentioned in video)
st.sidebar.header("Heatmap Configuration")
price_shock = st.sidebar.slider("Stock Price Shock Range", 0.1, 0.5, 0.3)
vol_shock = st.sidebar.slider("Volatility Shock Range", 0.2, 1.0, 0.5)

# Calculate basic prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Display basic results
col1, col2 = st.columns(2)
with col1:
    st.metric("Call Option Price", f"${call_price:.2f}")
with col2:
    st.metric("Put Option Price", f"${put_price:.2f}")

# Create heatmaps
if st.button("Generate Heatmaps"):
    S_range, sigma_range, call_prices, put_prices = create_heatmap_data(
        S, sigma, K, T, r, price_shock, vol_shock
    )
    
    # Create heatmap plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Call option heatmap
    sns.heatmap(call_prices, 
                xticklabels=np.round(S_range, 1)[::4], 
                yticklabels=np.round(sigma_range, 2)[::4],
                annot=False, 
                cmap='viridis', 
                ax=ax1)
    ax1.set_title('Call Option Prices')
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Volatility')
    
    # Put option heatmap
    sns.heatmap(put_prices, 
                xticklabels=np.round(S_range, 1)[::4], 
                yticklabels=np.round(sigma_range, 2)[::4],
                annot=False, 
                cmap='plasma', 
                ax=ax2)
    ax2.set_title('Put Option Prices')
    ax2.set_xlabel('Stock Price')
    ax2.set_ylabel('Volatility')
    
    plt.tight_layout()
    st.pyplot(fig)
