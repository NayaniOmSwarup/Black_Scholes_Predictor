#!/usr/bin/env python3
"""
Stage 5: Complete Application with MySQL Database Integration
Two-table structure: inputs and outputs tables
"""

import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from database_setup import setup_database, save_calculation, get_calculation_history, delete_calculation
import uuid

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
    """Create heatmap data for database storage"""
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

# Streamlit App Configuration
st.set_page_config(
    page_title="Black-Scholes with Database",
    page_icon="🗄️",
    layout="wide"
)

# Main title
st.title("🗄️ Black-Scholes Option Pricer with Database Storage")
st.markdown("*Complete implementation with MySQL two-table structure*")
st.markdown("---")

# Initialize database
if 'db_initialized' not in st.session_state:
    with st.spinner("Setting up database..."):
        st.session_state.db_initialized = setup_database()
    
    if st.session_state.db_initialized:
        st.success("✅ Database initialized successfully!")
    else:
        st.error("❌ Database initialization failed. Please check your MySQL configuration.")

# Sidebar for inputs
st.sidebar.header("🔧 Option Parameters")

# Core inputs (5 parameters for inputs table)
S = st.sidebar.number_input("Current Stock Price ($)", value=100.0, min_value=0.01, step=1.0)
K = st.sidebar.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=1.0)
T = st.sidebar.number_input("Time to Expiry (Years)", value=1.0, min_value=0.01, max_value=10.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility", value=0.20, min_value=0.01, max_value=5.0, step=0.01, format="%.4f")

# Heatmap configuration
st.sidebar.header("🔥 Heatmap Configuration")
price_shock = st.sidebar.slider("Stock Price Shock Range", 0.1, 0.5, 0.3, 0.05)
vol_shock = st.sidebar.slider("Volatility Shock Range", 0.2, 1.0, 0.5, 0.1)
grid_size = st.sidebar.selectbox("Grid Resolution", [15, 20, 25], index=1)

# Calculate current prices
call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Display current results
col1, col2 = st.columns(2)
with col1:
    st.metric("📞 Call Option Price", f"${call_price:.2f}")
with col2:
    st.metric("📉 Put Option Price", f"${put_price:.2f}")

st.markdown("---")

# Main functionality tabs
tab1, tab2, tab3 = st.tabs(["🎯 Calculate & Save", "📊 Calculation History", "🔥 Heatmap Analysis"])

with tab1:
    st.subheader("💾 Calculate and Save to Database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("""
        **Database Structure (as per video):**
        - **Inputs Table**: 6 columns (5 inputs + calculation_id)
        - **Outputs Table**: 4 main columns (volatility_shock, stock_price_shock, call_value, calculation_id)
        """)
    
    with col2:
        if st.button("💾 Save Calculation", type="primary", disabled=not st.session_state.db_initialized):
            with st.spinner("Generating heatmap data and saving to database..."):
                # Generate heatmap data for database storage
                heatmap_data = create_heatmap_data(S, sigma, K, T, r, price_shock, vol_shock, grid_size)
                
                # Save to database
                calculation_id = save_calculation(S, K, T, r, sigma, heatmap_data)
                
                if calculation_id:
                    st.success(f"✅ Calculation saved successfully!")
                    st.code(f"Calculation ID: {calculation_id}")
                    
                    # Show what was saved
                    S_range, sigma_range, call_prices, put_prices = heatmap_data
                    total_records = len(S_range) * len(sigma_range)
                    st.info(f"Saved {total_records} heatmap data points to outputs table")
                else:
                    st.error("❌ Failed to save calculation to database")

with tab2:
    st.subheader("📈 Calculation History")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Refresh History"):
            st.rerun()
    
    with col1:
        limit = st.selectbox("Number of records to show", [5, 10, 20, 50], index=1)
    
    if st.session_state.db_initialized:
        history = get_calculation_history(limit)
        
        if history:
            # Convert to DataFrame for better display
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Display as interactive table
            st.dataframe(
                df,
                column_config={
                    "calculation_id": "Calculation ID",
                    "stock_price": st.column_config.NumberColumn("Stock Price", format="$%.2f"),
                    "strike_price": st.column_config.NumberColumn("Strike Price", format="$%.2f"),
                    "time_to_expiry": st.column_config.NumberColumn("Time to Expiry", format="%.3f years"),
                    "risk_free_rate": st.column_config.NumberColumn("Risk-Free Rate", format="%.4f"),
                    "volatility": st.column_config.NumberColumn("Volatility", format="%.4f"),
                    "timestamp": st.column_config.DatetimeColumn("Timestamp")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Option to delete calculations
            st.subheader("🗑️ Delete Calculation")
            calc_ids = [row['calculation_id'] for row in history]
            selected_id = st.selectbox("Select calculation to delete", [""] + calc_ids)
            
            if selected_id and st.button("🗑️ Delete Selected", type="secondary"):
                if delete_calculation(selected_id):
                    st.success("✅ Calculation deleted successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete calculation")
        else:
            st.info("No calculations found in database")
    else:
        st.error("Database not available")

with tab3:
    st.subheader("🔥 Heatmap Visualization")
    
    if st.button("🎯 Generate Heatmaps", type="primary"):
        with st.spinner("Generating heatmaps..."):
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
            
            # Show data structure information
            st.info(f"""
            **Data Structure:**
            - Grid Size: {grid_size} x {grid_size} = {grid_size**2} data points
            - Stock Price Range: ${S_range[0]:.2f} to ${S_range[-1]:.2f}
            - Volatility Range: {sigma_range[0]:.3f} to {sigma_range[-1]:.3f}
            - This data would be stored in the outputs table with calculation_id linkage
            """)

# Footer with database info
st.markdown("---")
if st.session_state.db_initialized:
    st.markdown("🗄️ *Connected to MySQL database with two-table structure (inputs + outputs)*")
else:
    st.markdown("❌ *Database not connected - check MySQL configuration*")
