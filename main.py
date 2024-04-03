import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict


class ManualLR:
    def __init__(self, lag_features: Dict[str, Dict[str, int]], default_theta: float):
        """
        Initializes the ManualLR class.

        Parameters:
        - lag_features: A dictionary where keys are feature names and values are dictionaries
          containing 'lag': int for lag horizons and optionally 'theta': float for theta values.
        - default_theta: A default theta value to use for any feature not specifying its own theta.
        """
        self.lag_features = lag_features
        self.default_theta = default_theta

    def _create_modified_identity_matrix(self, size, start, end):
        """Create an identity matrix with modified diagonal values."""
        matrix = np.identity(size)
        diagonal_values = np.linspace(start, end, num=size)
        np.fill_diagonal(matrix, diagonal_values)
        logger.info(
            f"Created modified identity matrix of size {size} with start={start}, end={end}"
        )
        return matrix

    def _expand_matrix(self, matrix, new_rows, axis=0):
        """
        Expand the matrix with new rows or columns filled with zeros.
        """
        additional_shape = (
            (new_rows, matrix.shape[1]) if axis == 0 else (matrix.shape[0], new_rows)
        )
        expanded_matrix = np.concatenate(
            [matrix, np.zeros(additional_shape)], axis=axis
        )
        logger.info(f"Expanded matrix by {new_rows} rows/columns along axis {axis}")
        return expanded_matrix

    def preprocess_coef(self):
        """
        Preprocesses and returns the coefficient matrix based on the pre-configured parameters.
        Uses the 'lag_features' attribute to determine structure.
        """
        max_horizon = max(
            feature_info["lag"] for feature_info in self.lag_features.values()
        )
        coef_matrices = []
        for feature, info in self.lag_features.items():
            horizon = info["lag"]
            theta = info.get("theta", self.default_theta)
            ini_coef_matrix = self._create_modified_identity_matrix(
                horizon, theta, theta / 2
            )
            logger.info(
                f"Initial coefficient matrix for feature '{feature}' with theta={theta}"
            )

            if horizon < max_horizon:
                additional_rows = max_horizon - horizon
                expanded_coef_matrix = self._expand_matrix(
                    ini_coef_matrix, additional_rows, axis=0
                )
            else:
                expanded_coef_matrix = ini_coef_matrix

            coef_matrices.append(expanded_coef_matrix)

        final_matrix = np.concatenate(coef_matrices, axis=1)
        logger.info("Concatenated coefficient matrices for all features")
        return final_matrix

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data to create lagged feature DataFrame."""
        lagged_df = pd.DataFrame(index=df.index)
        for feature, info in self.lag_features.items():
            lag_horizon = info["lag"]
            for i in range(lag_horizon):
                lagged_df[f"{feature}_lag_{i+1}"] = df[feature].shift(i)
                logger.info(f"Generated lagged feature: {feature}_lag_{i+1}")

        non_na_lagged_df = lagged_df.dropna()
        return non_na_lagged_df

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict using the preprocessed coefficients and lagged features."""
        coef_matrix = self.preprocess_coef()
        lagged_df = self.preprocess_data(df)
        forecast_results = np.matmul(lagged_df.fillna(0).to_numpy(), coef_matrix.T)
        forecast_df = pd.DataFrame(
            forecast_results,
            index=lagged_df.index,
            columns=[f"Forecast_{i+1}" for i in range(coef_matrix.shape[0])],
        )
        logger.info("Generated forecast from processed data and coefficients")
        return forecast_df


def main():
    np.random.seed(0)  # For reproducible results
    data = {
        "sg": np.random.rand(100),
        "fr": np.random.rand(100),
    }
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame(data, index=dates)

    lag_features = {
        "sg": {"lag": 14, "theta": 0.012},
        "fr": {"lag": 5, "theta": 0.014},
    }
    model = ManualLR(lag_features, default_theta=0.015)
    forecast_df = model.predict(df)
    print(forecast_df.head())


if __name__ == "__main__":
    main()
