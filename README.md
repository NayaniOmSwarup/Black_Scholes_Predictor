# Black-Scholes Predictor

A robust, modular Python application for pricing European options using the Black-Scholes model. This project features an interactive Streamlit interface, heatmap visualizations for sensitivity and P&L analysis, and MySQL database integration for persistent calculation storage.

## Features

- **Black-Scholes Option Pricing:** Compute European call and put prices based on user-defined parameters.
- **Interactive Streamlit Interface:** User-friendly web app for inputting parameters and viewing results.
- **Heatmap Visualizations:** Analyze option price sensitivity to volatility and stock price changes with configurable heatmaps.
- **P&L Analysis:** Visualize profit and loss scenarios with intuitive color-coded heatmaps.
- **Database Integration:** Store calculation history and heatmap data in a normalized MySQL database.
- **Modular Structure:** Clean separation of pricing logic, database utilities, and Streamlit apps for easy maintenance and extension.

## Directory Structure

```
Black_Scholes_Predictor/
│
├── black_s.py               # Black-Scholes pricing logic (CLI)
├── config.py                # Database configuration
├── database_setup.py        # Database setup and utility functions
├── setup.py                 # Project setup script
├── streamlit_app.py         # Basic Streamlit GUI for option pricing
├── streamlit_with_pnl.py    # Streamlit app with P&L heatmaps
├── streamlit_full_app.py    # Full Streamlit app with database integration
└── requirements.txt         # Python dependencies
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NayaniOmSwarup/Black_Scholes_Predictor.git
   cd Black_Scholes_Predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up MySQL database:**
   - Install MySQL and create a database (e.g., `options_db`).
   - Update `config.py` with your database credentials.
   - Run the database setup script:
     ```bash
     python database_setup.py
     ```

## Usage

### Command-Line Pricing

```bash
python black_s.py
```
- Enter option parameters when prompted to receive call and put prices.

### Streamlit GUI (Basic)

```bash
streamlit run streamlit_app.py
```
- Web interface for pricing options interactively.

### Streamlit with P&L Heatmaps

```bash
streamlit run streamlit_with_pnl.py
```
- Visualize sensitivity and P&L heatmaps for various market scenarios.

### Full Streamlit App with Database

```bash
streamlit run streamlit_full_app.py
```
- All features plus persistent storage of calculations and heatmap data.

## Key Modules

| File                   | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `black_s.py`           | Black-Scholes pricing logic (call/put, CLI)                      |
| `streamlit_app.py`     | Basic Streamlit GUI for option pricing                           |
| `streamlit_with_pnl.py`| Streamlit GUI with P&L heatmaps                                  |
| `streamlit_full_app.py`| Full-featured Streamlit app with database integration            |
| `database_setup.py`    | Functions for creating and managing the MySQL database           |
| `config.py`            | Database configuration (host, user, password, database name)     |

## Example (Command-Line)

```
$ python black_s.py
=== Black-Scholes Option Pricer ===
Current stock price ($): 100
Strike price ($): 100
Time to expiry (years): 1
Risk-free interest rate (decimal, e.g., 0.05): 0.05
Volatility (decimal, e.g., 0.20): 0.2

RESULTS
Call Option Price: $10.45
Put Option Price:  $5.57
```

## Dependencies

- Python 3.8+
- streamlit
- numpy
- scipy
- matplotlib
- seaborn
- pandas
- mysql-connector-python

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Database Schema

- **Inputs Table:** Stores calculation parameters (stock price, strike price, time to expiry, risk-free rate, volatility, calculation ID, timestamp).
- **Outputs Table:** Stores heatmap data (volatility shock, stock price shock, call value, put value, calculation ID).

## Project Highlights

- Modular, extensible codebase for quantitative finance prototyping.
- Visual, interactive analysis for option pricing sensitivities and risk.
- Persistent storage for reproducible research and audit trails.
