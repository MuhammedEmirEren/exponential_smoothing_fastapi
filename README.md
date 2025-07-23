# 📈 E-Commerce Price Forecasting API

An advanced price prediction API using exponential smoothing algorithms, designed specifically for e-commerce applications. The API automatically selects the best forecasting model (Simple, Holt's, or Holt-Winters) based on your historical price data.

## 🚀 Features

- **Automatic Model Selection**: Compares Simple, Holt's, and Holt-Winters models
- **Real-time Forecasting**: Get instant price predictions
- **Visual Analytics**: Generate forecast charts
- **E-commerce Ready**: Perfect for pricing strategies and inventory planning
- **Easy Integration**: Simple REST API with JSON responses

## 📋 Requirements

```
Python 3.7+
FastAPI
Uvicorn
Pandas
NumPy
Matplotlib
Statsmodels
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MuhammedEmirEren/exponential_smoothing_fastapi.git
cd exponential_smoothing_fastapi
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the API server:**
```bash
python exponential_smoothing_API.py
```

The API will be available at: `http://127.0.0.1:8000`

## 📚 API Endpoints

### 1. **GET /** - API Documentation
Get basic information about the API and available endpoints.

**Response:**
```json
{
  "message": "Welcome to the Advanced Exponential Smoothing Time Series Forecasting API",
  "description": "Automatically selects the best model among Simple, Holt's, and Holt-Winters methods",
  "endpoints": {
    "/forecast": "Generate forecast using the best exponential smoothing model",
    "/plot": "Plot the forecast with historical data"
  }
}
```

### 2. **POST /forecast** - Get Price Forecast
Generate price predictions based on historical data.

**Request Body:**
```json
{
  "prices": [100, 102, 98, 105, 107, 103, 109, 111, 108, 115]
}
```

**Response:**
```json
{
  "forecast": [116.2, 117.1, 118.0, 118.8, 119.6, 120.4, 121.2, 122.0, 122.8, 123.6],
  "best_model": "holt",
  "rmse": 2.45,
  "input_data_points": 10
}
```

### 3. **POST /plot** - Generate Forecast Chart
Create a visual chart of the forecast with historical data.

**Request Body:**
```json
{
  "prices": [100, 102, 98, 105, 107, 103, 109, 111, 108, 115]
}
```

**Response:**
```json
{
  "message": "Plot generated successfully",
  "plot_saved": "exponential_smoothing_forecast.png",
  "forecast": [116.2, 117.1, 118.0, 118.8, 119.6, 120.4, 121.2, 122.0, 122.8, 123.6],
  "best_model": "holt",
  "rmse": 2.45,
  "training_points": 10,
  "forecast_points": 10
}
```

## 💻 Usage Examples
### Python
```python
import requests

def get_price_prediction(price_history):
    url = "http://127.0.0.1:8000/forecast"
    data = {"prices": price_history}
    
    response = requests.post(url, json=data)
    return response.json()

# Usage example
product_prices = [100, 102, 98, 105, 107, 103, 109]
prediction = get_price_prediction(product_prices)
print(f"Future price forecast: {prediction['forecast']}")
print(f"Model used: {prediction['best_model']}")
```

## 🎯 Models Explained

### Simple Exponential Smoothing
- **Best for**: Stable prices with no clear trend
- **Use case**: Products with consistent pricing
- **Output**: Flat forecast line

### Holt's Linear Trend Method
- **Best for**: Prices with clear upward/downward trends
- **Use case**: Growing or declining product categories
- **Output**: Trending forecast line

### Holt-Winters (Triple Exponential Smoothing)
- **Best for**: Prices with seasonal patterns
- **Use case**: Seasonal products, holiday items
- **Output**: Forecast with seasonal variations

## 📊 Response Fields Explained

| Field | Description |
|-------|-------------|
| `forecast` | Array of predicted future prices |
| `best_model` | The model that performed best on your data |
| `rmse` | Root Mean Square Error (lower is better) |
| `input_data_points` | Number of historical prices provided |
| `training_points` | Data points used for training |
| `forecast_points` | Number of future predictions |

## ⚠️ Important Notes

1. **Minimum Data**: Provide at least 3-5 historical price points for reliable forecasts
2. **Data Quality**: Ensure price data is clean and represents actual market prices
3. **Forecast Horizon**: Default forecast is 10 periods ahead (can be adjusted in code)
4. **Model Selection**: The API automatically selects the best model based on RMSE