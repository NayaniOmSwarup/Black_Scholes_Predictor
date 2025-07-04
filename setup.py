from setuptools import setup, find_packages

setup(
    name="black-scholes-pricer",
    version="1.0.0",
    description="Black-Scholes Option Pricer with Heatmap Visualizations",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "mysql-connector-python>=8.1.0"
    ],
    python_requires=">=3.8",
)
