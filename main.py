import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict
import mlflow.pyfunc


class ManualLR(mlflow.pyfunc.PythonModel):

    def __init__(self):
        self.params = None
        self.horizon = None

    def fit(self, params: Dict[str, Dict[str, int]]):
        """
        Fit.

        Parameters:
        - params: A dictionary where keys are feature names and values are dictionaries
          containing 'lag': int for lag horizons and optionally 'theta': float for theta values.
        """
        self.params = params

    def preprocess_coef(self):
        """
        Preprocesses and returns the coefficient matrix based on the pre-configured parameters.
        Uses the 'params' attribute to determine structure.
        """
        self.horizon = max(feature_info["lag"] for feature_info in self.params.values())
        coef_matrices = []
        for feature, info in self.params.items():
            feature_horizon = info.get("lag")
            theta = info.get("theta")
            temp_coef = np.identity(feature_horizon) * theta

            if feature_horizon < self.horizon:
                # extend to the longest horizon
                temp_coef = np.concatenate(
                    [
                        temp_coef,
                        np.zeros((self.horizon - feature_horizon, feature_horizon)),
                    ],
                    axis=0,
                )
            coef_matrices.append(temp_coef)

        final_matrix = np.concatenate(coef_matrices, axis=1)
        logger.info(
            f"Concatenated coefficient matrices for all features: {pd.DataFrame(final_matrix)}"
        )
        return final_matrix

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data to create lagged feature DataFrame."""
        lagged_df = pd.DataFrame(index=df.index)
        for feature, info in self.params.items():
            lag_horizon = info["lag"]
            for i in range(lag_horizon, 0, -1):
                lagged_df[f"{feature}_lag_{i}"] = df[feature].shift(i)
        logger.info(f"Generated lagged features: {lagged_df.columns.to_list()}")

        non_na_lagged_df = lagged_df.dropna()
        assert (
            not non_na_lagged_df.empty
        ), f"Not enough input data to forecast. Currently input lengh is {len(lagged_df)} but we should be equal to or more than {self.horizon}"
        return non_na_lagged_df

    def _column_sequence_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns of the DataFrame to match the sequence defined in self.params.
        It constructs expected lagged feature names based on self.params and orders df accordingly.

        Parameters:
        - df: Input DataFrame with columns that need to be reordered.

        Returns:
        - reordered_df: DataFrame with columns reordered to match the keys sequence in self.params.
        """
        reordered_cols = self.params.keys()
        reordered_df = df[reordered_cols]

        logger.info("Columns reordered to match the expected sequence.")
        return reordered_df

    def predict(self, context, df: pd.DataFrame) -> pd.DataFrame:
        """Predict using the preprocessed coefficients and lagged features."""

        df = self._column_sequence_check(df)

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
    data_len = 100
    data = {
        "fruit": np.random.rand(data_len),
        "green": np.random.rand(data_len),
        "fruitset": np.random.rand(data_len),
    }
    dates = pd.date_range(start="2023-01-01", periods=data_len)
    df = pd.DataFrame(data, index=dates)

    params = {
        "fruit": {"lag": 5, "theta": 0.014},
        "green": {"lag": 14, "theta": 0.012},
        "fruitset": {"lag": 21, "theta": 0.001},
    }
    model = ManualLR()
    model.fit(params)
    forecast_df = model.predict(None, df)
    print(forecast_df.head())


if __name__ == "__main__":
    main()
