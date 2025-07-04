import numpy as np
from scipy.stats import norm

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

def main():
    """Simple REPL interface for option pricing"""
    print("=== Black-Scholes Option Pricer ===")
    
    # Get the five core inputs
    S = float(input("Enter current stock price: "))
    K = float(input("Enter strike price: "))
    T = float(input("Enter time to expiry (years): "))
    r = float(input("Enter risk-free interest rate (decimal): "))
    sigma = float(input("Enter volatility (decimal): "))
    
    # Calculate and display both call and put prices
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    
    print(f"\nCall Option Price: ${call_price:.2f}")
    print(f"Put Option Price: ${put_price:.2f}")

if __name__ == "__main__":
    main()
