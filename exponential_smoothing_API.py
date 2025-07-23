from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from fastapi import FastAPI
from pydantic import BaseModel
warnings.filterwarnings('ignore')
from typing import List

app = FastAPI()
# API for Exponential Smoothing Time Series Forecasting

class PriceHistory(BaseModel):
    prices: List[float]

@app.get("/")
def readme():
    """
    Readme for the Exponential Smoothing Time Series Forecasting API.
    """
    return {
        "message": "Welcome to the Advanced Exponential Smoothing Time Series Forecasting API",
        "description": "Automatically selects the best model among Simple, Holt's, and Holt-Winters methods",
        "endpoints": {
            "/forecast": "Generate forecast using the best exponential smoothing model (auto-selected)",
            "/plot": "Plot the forecast with historical data using the best model"
        },
        "models_available": {
            "simple": "Simple Exponential Smoothing (level only)",
            "holt": "Holt's Linear Trend method (level + trend)",
            "holtwinters": "Holt-Winters method (level + trend + seasonality)"
        }
    }

@app.post("/forecast")
def forecast(history: PriceHistory):
    """
    Generate forecast using the best Exponential Smoothing model.
    """
    try:
        data = pd.Series(history.prices)

        # Compare models and select the best one
        model_results = compare_exponential_smoothing_models(data)
        best_model_type = None
        best_rmse = float('inf')
        
        for model_type, result in model_results.items():
            if result and result['rmse'] < best_rmse:
                best_rmse = result['rmse']
                best_model_type = model_type
        
        if best_model_type:
            print(f"\nBest model: {best_model_type.title()} (RMSE: {best_rmse:.2f})")
            best_model = model_results[best_model_type]['model']
        else:
            print("Using Holt's method as fallback")
            best_model = create_exponential_smoothing_model(data, 'holt')
            best_model_type = 'holt'

        # Generate forecast using the best model
        forecast = forecast_simple_exponential_smoothing(best_model, steps=10)
        
        return {
            "forecast": forecast.tolist(),
            "best_model": best_model_type,
            "rmse": best_rmse if best_model_type else "N/A",
            "input_data_points": len(data)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/plot")
def plot(history: PriceHistory):
    """
    Plot the forecast with historical data using the best model.
    """
    try:
        data = pd.Series(history.prices)
        
        # Compare models and select the best one
        model_results = compare_exponential_smoothing_models(data)
        best_model_type = None
        best_rmse = float('inf')
        
        for model_type, result in model_results.items():
            if result and result['rmse'] < best_rmse:
                best_rmse = result['rmse']
                best_model_type = model_type
        
        if best_model_type:
            best_model = model_results[best_model_type]['model']
        else:
            best_model = create_exponential_smoothing_model(data, 'holt')
            best_model_type = 'holt'
        
        # Generate forecast
        forecast = forecast_simple_exponential_smoothing(best_model, steps=10)
        
        # Create the plot
        plot_exponential_smoothing_forecast(data, forecast)
        
        return {
            "message": "Plot generated successfully",
            "plot_saved": "exponential_smoothing_forecast.png",
            "forecast": forecast.tolist(),
            "best_model": best_model_type,
            "rmse": best_rmse if best_model_type else "N/A",
            "training_points": len(data),
            "forecast_points": len(forecast)
        }
    except Exception as e:
        return {"error": str(e)}

def compare_exponential_smoothing_models(data):
    """Compare different exponential smoothing models."""
    models = ['simple', 'holt', 'holtwinters']
    results = {}
    
    print(f"Comparing models for {len(data)} data points...")
    
    for model_type in models:
        try:
            print(f"Testing {model_type} model...")
            model = create_exponential_smoothing_model(data, model_type)
            
            # Calculate AIC for comparison
            aic = model.aic if hasattr(model, 'aic') else np.nan
            
            # Calculate RMSE on training data
            fitted_values = model.fittedvalues
            if len(fitted_values) == len(data):
                rmse = np.sqrt(np.mean((data - fitted_values) ** 2))
            else:
                # Handle cases where fitted values might be shorter
                min_len = min(len(data), len(fitted_values))
                rmse = np.sqrt(np.mean((data[-min_len:] - fitted_values[-min_len:]) ** 2))
            
            results[model_type] = {
                'model': model,
                'aic': aic,
                'rmse': rmse
            }
            
            print(f"{model_type.title()} - AIC: {aic:.2f}, RMSE: {rmse:.2f}")
            
        except Exception as e:
            print(f"Failed to fit {model_type} model: {e}")
            results[model_type] = None
    
    return results

def create_exponential_smoothing_model(data, model_type='holt'):
    """
    Create and fit an Exponential Smoothing model.

    Parameters:
    - data: Time series data
    - model_type: 'simple', 'holt', or 'holtwinters'

    Returns:
    - fitted_model: Fitted Exponential Smoothing model.
    """
    if model_type == 'simple':
        print("Fitting Simple Exponential Smoothing...")
        model = SimpleExpSmoothing(data)
        fitted_model = model.fit()
    elif model_type == 'holt':
        print("Fitting Holt's Linear Trend method...")
        model = Holt(data)
        fitted_model = model.fit()
    elif model_type == 'holtwinters':
        print("Fitting Holt-Winters (Triple Exponential Smoothing)...")
        # Try seasonal model, fallback to trend-only if it fails
        try:
            model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=min(12, len(data)//2))
            fitted_model = model.fit()
        except:
            print("Seasonal model failed, using trend-only model...")
            model = ExponentialSmoothing(data, trend='add')
            fitted_model = model.fit()
    else:
        # Default to Holt's method
        print("Unknown model type, using Holt's method...")
        model = Holt(data)
        fitted_model = model.fit()
    
    return fitted_model

def forecast_simple_exponential_smoothing(model, steps=30):
    """
    Generate forecast using the fitted Simple Exponential Smoothing model.

    Parameters:
    - model: Fitted Simple Exponential Smoothing model.
    - steps: Number of periods to forecast.

    Returns:
    - forecast: Series containing the forecasted values.
    - conf_int: Confidence intervals (simulated for exponential smoothing)
    """
    forecast = model.forecast(steps=steps)
    return forecast

def plot_exponential_smoothing_forecast(data, forecast):
    """
    Plot the Simple Exponential Smoothing forecast with historical data.

    Parameters:
    - data: Historical time series data.
    - forecast: Series containing the forecasted values.
    """
    plt.figure(figsize=(12, 8))

    # Plot historical data
    plt.plot(data.index, data, 'o-', 
             label='Historical Data', color='blue', linewidth=2, markersize=3)
    
    # Create forecast indices (continuing from the end of historical data)
    forecast_start_index = data.index[-1] + 1
    forecast_indices = range(forecast_start_index, forecast_start_index + len(forecast))
    
    # Plot forecast
    plt.plot(forecast_indices, forecast, '--', 
             label='Forecast', color='red', linewidth=2)
    
    # Add separation line
    plt.axvline(x=data.index[-1], color='green', linestyle='--', linewidth=2,
                label='Historical Data | Forecast Boundary')
    
    plt.title('Exponential Smoothing Price Prediction Forecast')
    plt.xlabel('Time Period')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('exponential_smoothing_forecast.png')
    plt.close()  # Close the figure to free memory
    
    return 'exponential_smoothing_forecast.png'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
