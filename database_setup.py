#!/usr/bin/env python3
"""
Database setup and utility functions
"""

import mysql.connector
from mysql.connector import Error
import uuid
from typing import Tuple, Optional, List
from config import DATABASE_CONFIG, INPUTS_TABLE_SCHEMA, OUTPUTS_TABLE_SCHEMA

def create_database_connection():
    """Create database connection"""
    try:
        connection = mysql.connector.connect(**DATABASE_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def setup_database():
    """Setup database with two tables as shown in video"""
    connection = create_database_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        # Create inputs table (6 columns: 5 inputs + calculation_id)
        cursor.execute(INPUTS_TABLE_SCHEMA)
        
        # Create outputs table (4 main columns: vol_shock, price_shock, call_value, calculation_id)
        cursor.execute(OUTPUTS_TABLE_SCHEMA)
        
        connection.commit()
        print("Database tables created successfully!")
        return True
        
    except Error as e:
        print(f"Error setting up database: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def save_calculation(S: float, K: float, T: float, r: float, sigma: float, 
                    heatmap_data: Tuple) -> Optional[str]:
    """Save calculation to database and return calculation ID"""
    
    calculation_id = str(uuid.uuid4())
    connection = create_database_connection()
    
    if not connection:
        return None
    
    try:
        cursor = connection.cursor()
        
        # Insert into inputs table
        inputs_query = """
        INSERT INTO inputs (calculation_id, stock_price, strike_price, time_to_expiry, risk_free_rate, volatility)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(inputs_query, (calculation_id, S, K, T, r, sigma))
        
        # Unpack heatmap data
        S_range, sigma_range, call_prices, put_prices = heatmap_data
        
        # Insert heatmap data into outputs table
        outputs_query = """
        INSERT INTO outputs (calculation_id, volatility_shock, stock_price_shock, call_value, put_value)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        # Insert each combination from the heatmap
        for i, vol in enumerate(sigma_range):
            for j, price in enumerate(S_range):
                cursor.execute(outputs_query, (
                    calculation_id, 
                    float(vol), 
                    float(price), 
                    float(call_prices[i, j]), 
                    float(put_prices[i, j])
                ))
        
        connection.commit()
        return calculation_id
        
    except Error as e:
        print(f"Error saving calculation: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def get_calculation_history(limit: int = 10) -> List[dict]:
    """Retrieve calculation history"""
    connection = create_database_connection()
    if not connection:
        return []
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT calculation_id, stock_price, strike_price, time_to_expiry, 
               risk_free_rate, volatility, timestamp
        FROM inputs 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        return results
        
    except Error as e:
        print(f"Error retrieving history: {e}")
        return []
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def delete_calculation(calculation_id: str) -> bool:
    """Delete a calculation and its associated outputs"""
    connection = create_database_connection()
    if not connection:
        return False
    
    try:
        cursor = connection.cursor()
        
        # Delete from inputs (outputs will be deleted automatically due to CASCADE)
        cursor.execute("DELETE FROM inputs WHERE calculation_id = %s", (calculation_id,))
        connection.commit()
        
        return cursor.rowcount > 0
        
    except Error as e:
        print(f"Error deleting calculation: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    # Test database setup
    if setup_database():
        print("Database setup completed successfully!")
    else:
        print("Database setup failed!")
