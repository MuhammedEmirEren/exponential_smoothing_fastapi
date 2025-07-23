from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from fastapi import FastAPI
warnings.filterwarnings('ignore')

app = FastAPI()
# API for Exponential Smoothing Time Series Forecasting
@app.get("/")
def readme():
    """
    Readme for the Exponential Smoothing Time Series Forecasting API.
    """
    return {
        "message": "Welcome to the Exponential Smoothing Time Series Forecasting API",
        "endpoints": {
            "/forecast": "Generate forecast using Simple Exponential Smoothing",
            "/plot": "Plot the forecast with historical data"
        }
    }

@app.get("/forecast")
def forecast():
    """
    Generate forecast using Simple Exponential Smoothing.
    This endpoint is a placeholder for the actual forecasting logic.
    """
    return {"message": "Forecasting endpoint is under construction."}
@app.get("/plot")
def plot():
    """
    Plot the forecast with historical data.
    This endpoint is a placeholder for the actual plotting logic.
    """
    return {"message": "Plotting endpoint is under construction."}

def create_simple_exponential_smoothing_model(data):
    """
    Create and fit a Simple Exponential Smoothing model.

    Parameters:
    - data: DataFrame containing the time series data.

    Returns:
    - fitted_model: Fitted Simple Exponential Smoothing model.
    """
    model = SimpleExpSmoothing(data)
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
    
    # Simple exponential smoothing doesn't provide confidence intervals directly
    # We'll create a simple approximation based on the residuals
    residuals = model.resid
    std_error = np.std(residuals)
    
    # Create confidence intervals (approximate)
    lower_bound = forecast - 1.96 * std_error  # 95% CI
    upper_bound = forecast + 1.96 * std_error
    
    # Create a DataFrame similar to what other models return
    conf_int = pd.DataFrame({
        'lower': lower_bound,
        'upper': upper_bound
    })
    
    return forecast, conf_int

def plot_exponential_smoothing_forecast(data, training_data, model, forecast, conf_int, test_data, dates, future_dates):
    """
    Plot the Simple Exponential Smoothing forecast with historical data.

    Parameters:
    - data: Historical time series data.
    - model: Fitted Simple Exponential Smoothing model.
    - forecast: Series containing the forecasted values.
    - dates: Dates corresponding to the historical data.
    - future_dates: Dates for the forecasted values.
    """
    plt.figure(figsize=(12, 8))

    # Plot training data
    plt.plot(training_data['ds'], training_data['y'], 'o-', 
             label='Training Data (Actual)', color='blue', linewidth=2, markersize=3)
    
    # Plot actual test data
    plt.plot(test_data['ds'], test_data['y'], 'o-', 
             label='Test Data (Actual)', color='green', linewidth=2, markersize=4)
    
    # Plot forecast
    plt.plot(test_data['ds'], forecast, '--', 
             label='SARIMA Predictions', color='red', linewidth=2)
    

    # Plot confidence intervals
    plt.fill_between(future_dates, conf_int['lower'], conf_int['upper'], 
                     color='red', alpha=0.3, label='95% Confidence Interval')
    
    # Add separation line
    last_historical_date = training_data['ds'].max()
    plt.axvline(x=last_historical_date, color='green', linestyle='--', linewidth=2,
                label='Historical Data | Prediction Boundary')
    
    plt.title('Exponential Smoothing Price Prediction Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def compute_mean_squared_error_of_model(test_data, forecast, train_size):
    """
    Compute the Mean Squared Error (MSE) of the model's predictions.

    Parameters:
    - test_data: Actual test data.
    - forecast: Forecasted values (Series for Exponential Smoothing).
    - train_size: Size of the training data (not used for Exponential Smoothing).

    Returns:
    - mape: Mean Absolute Percentage Error.
    """
    # For Exponential Smoothing, forecast is already the predictions for test period
    test_predictions = forecast.values
    actual_test = test_data['y'].values
    
    if len(test_predictions) != len(actual_test):
        min_len = min(len(test_predictions), len(actual_test))
        test_predictions = test_predictions[:min_len]
        actual_test = actual_test[:min_len]
    
    # Check for NaN values and print debugging info
    if np.isnan(test_predictions).any():
        print("WARNING: NaN values found in predictions!")
        print(f"Predictions: {test_predictions[:10]}")  # Show first 10 values
    
    if np.isnan(actual_test).any():
        print("WARNING: NaN values found in actual test data!")
        print(f"Actual: {actual_test[:10]}")  # Show first 10 values
    
    mse = ((actual_test - test_predictions) ** 2).mean()
    
    print(f"\n=== Exponential Smoothing Model Performance ===")
    print(f"Test data points: {len(actual_test)}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {mse**0.5:.2f}")
    print(f"Mean Absolute Error: {abs(actual_test - test_predictions).mean():.2f}")
    
    # Calculate percentage error
    mape = abs((actual_test - test_predictions) / actual_test).mean() * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

    return mape
def main():
    averageErrorRate = 0.0
    for i in range(1, 11):
        # Load the data for each file
        print(f"Processing file: price_history_{i}.csv")
        # Load the data
        data = pd.read_csv(f'price_history_{i}.csv') 
        data['ds'] = pd.to_datetime(data['ds'])
        data = data.sort_values('ds').reset_index(drop=True)

        # Split data: last 90 days as test, rest as training
        test_size = 90
        if len(data) < test_size:
            print(f"Warning: Data has only {len(data)} points, using last {len(data)//2} as test data")
            test_size = len(data) // 2
    
        split_point = len(data) - test_size
        training_data = data[:split_point][['ds', 'y']].copy()
        test_data = data[split_point:][['ds', 'y']].copy()
    
        print(f"Total data points: {len(data)}")
        print(f"Training data: {len(training_data)} points ({data['ds'].iloc[0]} to {training_data['ds'].iloc[-1]})")
        print(f"Test data: {len(test_data)} points ({test_data['ds'].iloc[0]} to {test_data['ds'].iloc[-1]})")

        # Create and fit the model on training data only
        model = create_simple_exponential_smoothing_model(training_data['y'])

        # Generate forecast for the test period
        forecast, conf_int = forecast_simple_exponential_smoothing(model, steps=len(test_data))

        # Calculate and display performance metrics
        SingleMape = compute_mean_squared_error_of_model(test_data, forecast, len(training_data))
        averageErrorRate += SingleMape
        

    
    averageErrorRate = averageErrorRate / 10
    print(f"\nAverage Mean Absolute Percentage Error across all files: {averageErrorRate:.2f}%")

    # Plot the forecast with separation line showing actual test data
    fig = plot_exponential_smoothing_forecast(data, training_data, model, forecast, conf_int,
                                            test_data, training_data['ds'], 
                                            test_data['ds'] + pd.to_timedelta(np.arange(len(forecast)), unit='D'))
    fig.show()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
