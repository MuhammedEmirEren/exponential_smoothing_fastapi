from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
warnings.filterwarnings('ignore')
from typing import List
import numpy as np

# Suppress Prophet's plotly warning
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)

app = FastAPI()
# API for Exponential Smoothing Time Series Forecasting

class PriceHistory(BaseModel):
    prices: List[float]

@app.get("/")
def readme():
    """
    Readme for the Prophet Time Series Forecasting API.
    """
    return {
        "message": "Welcome to the Advanced Prophet Time Series Forecasting API",
        "description": "Forecasts time series data using the Prophet model",
        "endpoints": {
            "/forecast": "Generate forecast using the Prophet model"
        }
    }

@app.post("/forecast")
def forecast(history: PriceHistory):
    """
    Generate forecast using the best Exponential Smoothing model.
    """
    try:
        data = pd.DataFrame({'y': history.prices})
        data['ds'] = pd.date_range(start='2025-01-01', periods=len(data), freq='D')
        data = data[['ds', 'y']]
        data['y'] = data['y'].astype(float)

        model = create_prophet_model(data)

        # Generate forecast for the test period
        forecast = forecast_prophet(model, periods=10)
        
        # Extract just the forecasted values (the new predictions)
        forecast_values = forecast['yhat'].tail(10).values
        
        return {
            "forecast": forecast_values.tolist(),
            "confidence_interval": {
                "lower": forecast['yhat_lower'].tail(10).values.tolist(),
                "upper": forecast['yhat_upper'].tail(10).values.tolist()
            },
            "model_type": "Prophet",
            "input_data_points": len(data),
            "forecast_periods": 10
        }
    except Exception as e:
        return {"error": str(e)}



def create_prophet_model(data, growth='linear', seasonality_mode='additive'):
    """
    Create and fit a Prophet model with specified growth and seasonality mode.

    Parameters:
    - data: DataFrame containing the time series data with 'ds' and 'y' columns.
    - growth: Type of growth ('linear' or 'logistic').
    - seasonality_mode: Type of seasonality ('additive' or 'multiplicative').

    Returns:
    - fitted_model: Fitted Prophet model.
    """
    model = Prophet(growth=growth, seasonality_mode=seasonality_mode)
    model.fit(data)
    return model

def forecast_prophet(model, periods=30):
    """
    Generate forecast using the fitted Prophet model.

    Parameters:
    - model: Fitted Prophet model.
    - periods: Number of periods to forecast.

    Returns:
    - forecast: DataFrame containing the forecasted values.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_forecast(model, forecast, data, training_data):
    """
    Plot the forecasted values with a line separating training data from predictions.

    Parameters:
    - model: Fitted Prophet model.
    - forecast: DataFrame containing the forecasted values.
    - data: Full dataset including test data.
    - training_data: Training data to determine separation point.
    """
    # Create custom plot to show both training and test data clearly
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot training data
    ax.plot(training_data['ds'], training_data['y'], 'o-', 
            label='Training Data (Actual)', color='blue', linewidth=2, markersize=3)
    
    # Plot test data (actual values)
    test_data = data[len(training_data):].copy()
    ax.plot(test_data['ds'], test_data['y'], 'o-', 
            label='Test Data (Actual)', color='green', linewidth=2, markersize=4)
    
    # Plot forecast for the test period
    forecast_test = forecast.iloc[len(training_data):len(training_data) + len(test_data)]
    ax.plot(forecast_test['ds'], forecast_test['yhat'], '--', 
            label='Test Predictions', color='red', linewidth=2)
    
    # Plot confidence intervals for test predictions
    ax.fill_between(forecast_test['ds'], 
                    forecast_test['yhat_lower'], 
                    forecast_test['yhat_upper'], 
                    alpha=0.3, color='red', label='Prediction Confidence Interval')
    
    # Add vertical line to separate training from test
    last_training_date = training_data['ds'].max()
    ax.axvline(x=last_training_date, color='black', linestyle='--', linewidth=2, 
               label='Training | Test Split')
    
    ax.set_title('Prophet Model: Training vs Test Period with Actual Test Data', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return fig

def compute_mean_squared_error_of_model(test_data, forecast, train_size):
    """
    Compute the Mean Squared Error between actual test data and predicted values.

    Parameters:
    - test_data: DataFrame with actual test values (ds, y columns).
    - forecast: Prophet forecast DataFrame.
    - train_size: Number of training data points.

    Returns:
    - mse: Mean Squared Error.
    """
    # Get predictions for the test period (after training data)
    test_predictions = forecast['yhat'].iloc[train_size:train_size + len(test_data)].values
    actual_test = test_data['y'].values
    
    if len(test_predictions) != len(actual_test):
        min_len = min(len(test_predictions), len(actual_test))
        test_predictions = test_predictions[:min_len]
        actual_test = actual_test[:min_len]
    
    mse = ((actual_test - test_predictions) ** 2).mean()
    
    print(f"\n=== Model Performance ===")
    print(f"Test data points: {len(actual_test)}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {mse**0.5:.2f}")
    print(f"Mean Absolute Error: {abs(actual_test - test_predictions).mean():.2f}")
    
    # Calculate percentage error
    mean_actual = actual_test.mean()
    mape = abs((actual_test - test_predictions) / actual_test).mean() * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    return mape

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
