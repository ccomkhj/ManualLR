# ManualLR: A Handcrafted Matrix Multiplication-based Forecasting Model

This Python script introduces the ManualLR (Manual Linear Regression) class, a customizable tool designed to forecast time series data using a matrix multiplication approach. It stands out for its ability to handle feature-specific lag configurations and varying theta values, offering a flexible and intuitive way to perform forecasting on time series datasets.

## Features

- **Custom Lag and Theta Values**: Define the lag (historical horizon) and optional theta (weight decay parameter) for each feature individually, allowing for tailored preprocessing according to the significance and temporal reach of each feature.
- **Dynamic Coefficient Matrix Creation**: Generates a unique coefficient matrix per feature based on the specified lag and theta values, which affects both the weight and the extent of historical data considered in the prediction.
- **Matrix Expansion and Concatenation**: Seamlessly expands smaller matrices to align with the feature requiring the maximum lag, ensuring consistent dimensions across all features for proper matrix multiplication.
- **Lagged Feature Extraction**: Automatically produces lagged versions of each feature up to the defined horizon, enabling the model to leverage historical data for prediction.
- **Simplified Forecast Generation**: Combines the preprocessed data and coefficients in a single step through matrix multiplication, outputting a forecast for