#!/usr/bin/env python3
"""
Database configuration settings
"""

import os
from typing import Dict

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'database': os.getenv('DB_NAME', 'options_db'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# Table schemas as described in video
INPUTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS inputs (
    calculation_id VARCHAR(36) PRIMARY KEY,
    stock_price DECIMAL(10,2) NOT NULL,
    strike_price DECIMAL(10,2) NOT NULL,
    time_to_expiry DECIMAL(5,4) NOT NULL,
    risk_free_rate DECIMAL(5,4) NOT NULL,
    volatility DECIMAL(5,4) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
"""

OUTPUTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS outputs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    calculation_id VARCHAR(36) NOT NULL,
    volatility_shock DECIMAL(5,4) NOT NULL,
    stock_price_shock DECIMAL(10,2) NOT NULL,
    call_value DECIMAL(10,4) NOT NULL,
    put_value DECIMAL(10,4) NOT NULL,
    FOREIGN KEY (calculation_id) REFERENCES inputs(calculation_id) ON DELETE CASCADE
)
"""
