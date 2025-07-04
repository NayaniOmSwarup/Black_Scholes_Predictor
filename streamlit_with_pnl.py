#!/usr/bin/env python3
"""
Stage 4: Enhanced P&L Analysis
Adds profit/loss analysis with color coding (green/red)
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

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
    """Create heatmap data for option prices"""
    S_min = base_S * (1 - price_range)
    S_max = base_S * (1 + price_range)
    sigma_min = max(0.01, base_sigma * (1 - vol_range))
    sigma_max = base_sigma * (1 + vol_range)
    
    S_range = np.linspace(S_min, S_max, grid_size)
    sigma_range = np.linspace(sigma_min, sigma_max, grid_size)
    
    call_prices = np.zeros((len(sigma_range), len(S_range)))
    put_prices = np.zeros((len(sigma_range), len(S_range)))
    
    for i, vol in enumerate(sigma_range):
        for j, price in enumerate(S_range):
            call_prices[i, j] = black_scholes_call(price, K, T, r, vol)
            put_prices[i, j] = black_scholes_put(price, K, T, r, vol)
    
    return S_range, sigma_range, call_prices, put_prices

def create_pnl_heatmap_data(base_S, base_sigma, K, T, r, call_buy_price, put_buy_price, 
                           price_range=0.3, vol_range=0.5, grid_size=20):
    """Create P&L heatmap data"""
    S_range, sigma_range, call_prices, put_prices = create_heatmap_data(
        base_S, base_sigma, K, T, r, price_range, vol_range, grid_size
    )
    
    # Calculate P&L by subtracting purchase prices
    call_pnl = call_prices - call_buy_price
    put_pnl = put_prices - put_buy_price
    
    return S_range, sigma_range, call_pnl, put_pnl

# Streamlit App Configuration
st.set_page_config(
    page_title="Black-Scholes P&L Analysis",
    page_icon="💰",
    layout="wide"
)

# Main title
st.title("💰 Black-Scholes Option Pricer with P&L Analysis")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("🔧 Option Parameters")

# Core inputs
S = st.sidebar.number_input("Current Stock Price ($)", value=100.0, min_value=0.01, step=1.0)
K = st.sidebar.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=1.0)
T = st.sidebar.number_input("Time to Expiry (Years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility", value=0.20, min_value=0.01, max_value=5.0, step=0.01, format="%.4f")

# P&L Analysis inputs (new in Stage 4)
st.sidebar.header("💰 P&L Analysis")
st.sidebar.markdown("Enter purchase prices for P&L calculation:")

call_purchase_price = st.sidebar.number_input(
    "Call Purchase Price ($)", 
    value=10.0, 
    min_value=0.0, 
    step=0.5,
    help="Price you paid for the call option"
)

put_purchase_price = st.sidebar.number_input(
    "Put Purchase Price ($)", 
    value=5.0, 
    min_value=0.0, 
    step=0.5,
    help="Price you paid for the put option"
)

# Heatmap configuration
st.sidebar.header("🔥 Heatmap Configuration")
price_shock = st.sidebar.slider("Stock Price Shock Range", 0.1, 0.5, 0.3, 0.05)
vol_shock = st.sidebar.slider("Volatility Shock Range", 0.2, 1.0, 0.5, 0.1)
grid_size = st.sidebar.selectbox("Grid Resolution", [15, 20, 25, 30], index=1)

# Calculate current option prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Calculate current P&L
current_call_pnl = call_price - call_purchase_price
current_put_pnl = put_price - put_purchase_price

# Display current results
st.subheader("📊 Current Option Values & P&L")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📞 Call Price", f"${call_price:.2f}")
with col2:
    st.metric("📉 Put Price", f"${put_price:.2f}")
with col3:
    delta_call = current_call_pnl
    st.metric("📈 Call P&L", f"${current_call_pnl:.2f}", delta=f"${delta_call:.2f}")
with col4:
    delta_put = current_put_pnl
    st.metric("📊 Put P&L", f"${current_put_pnl:.2f}", delta=f"${delta_put:.2f}")

st.markdown("---")

# Heatmap generation
tab1, tab2 = st.tabs(["🔥 Option Price Heatmaps", "💰 P&L Heatmaps"])

with tab1:
    if st.button("🎯 Generate Price Heatmaps", type="primary"):
        with st.spinner("Generating option price heatmaps..."):
            S_range, sigma_range, call_prices, put_prices = create_heatmap_data(
                S, sigma, K, T, r, price_shock, vol_shock, grid_size
            )
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Call option heatmap
            sns.heatmap(call_prices, 
                       xticklabels=np.round(S_range, 1)[::max(1, len(S_range)//8)], 
                       yticklabels=np.round(sigma_range, 3)[::max(1, len(sigma_range)//8)],
                       annot=False, cmap='viridis', ax=ax1, cbar_kws={'label': 'Call Price ($)'})
            ax1.set_title('Call Option Prices', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Stock Price ($)')
            ax1.set_ylabel('Volatility')
            
            # Put option heatmap
            sns.heatmap(put_prices, 
                       xticklabels=np.round(S_range, 1)[::max(1, len(S_range)//8)], 
                       yticklabels=np.round(sigma_range, 3)[::max(1, len(sigma_range)//8)],
                       annot=False, cmap='plasma', ax=ax2, cbar_kws={'label': 'Put Price ($)'})
            ax2.set_title('Put Option Prices', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Stock Price ($)')
            ax2.set_ylabel('Volatility')
            
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    if st.button("💰 Generate P&L Heatmaps", type="primary"):
        with st.spinner("Generating P&L heatmaps with color coding..."):
            S_range, sigma_range, call_pnl, put_pnl = create_pnl_heatmap_data(
                S, sigma, K, T, r, call_purchase_price, put_purchase_price, 
                price_shock, vol_shock, grid_size
            )
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Call P&L heatmap with green/red color coding
            sns.heatmap(call_pnl, 
                       xticklabels=np.round(S_range, 1)[::max(1, len(S_range)//8)], 
                       yticklabels=np.round(sigma_range, 3)[::max(1, len(sigma_range)//8)],
                       annot=False, 
                       cmap='RdYlGn',  # Red for negative, Green for positive
                       center=0,       # Center colormap at zero
                       ax=ax1, 
                       cbar_kws={'label': 'Call P&L ($)'})
            ax1.set_title('Call Option P&L (Green=Profit, Red=Loss)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Stock Price ($)')
            ax1.set_ylabel('Volatility')
            
            # Put P&L heatmap with green/red color coding
            sns.heatmap(put_pnl, 
                       xticklabels=np.round(S_range, 1)[::max(1, len(S_range)//8)], 
                       yticklabels=np.round(sigma_range, 3)[::max(1, len(sigma_range)//8)],
                       annot=False, 
                       cmap='RdYlGn',  # Red for negative, Green for positive
                       center=0,       # Center colormap at zero
                       ax=ax2, 
                       cbar_kws={'label': 'Put P&L ($)'})
            ax2.set_title('Put Option P&L (Green=Profit, Red=Loss)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Stock Price ($)')
            ax2.set_ylabel('Volatility')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # P&L Summary
            st.subheader("📈 P&L Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Call Profit", f"${np.max(call_pnl):.2f}")
            with col2:
                st.metric("Max Call Loss", f"${np.min(call_pnl):.2f}")
            with col3:
                st.metric("Max Put Profit", f"${np.max(put_pnl):.2f}")
            with col4:
                st.metric("Max Put Loss", f"${np.min(put_pnl):.2f}")
            
            # Profit probability
            call_profit_pct = (call_pnl > 0).sum() / call_pnl.size * 100
            put_profit_pct = (put_pnl > 0).sum() / put_pnl.size * 100
            
            st.info(f"""
            **Profitability Analysis:**
            - Call option profitable in **{call_profit_pct:.1f}%** of scenarios
            - Put option profitable in **{put_profit_pct:.1f}%** of scenarios
            """)

# Footer
st.markdown("---")
st.markdown("*Enhanced with P&L analysis and color-coded visualizations*")
