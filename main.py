import numpy as np
import pandas as pd


class HexaLinearRegression:
    def __init__(self, horizon_sg, horizon_fr, theta_sg, theta_fr):
        self.horizon_sg = horizon_sg
        self.horizon_fr = horizon_fr
        self.theta_sg = theta_sg
        self.theta_fr = theta_fr

    def _create_modified_identity_matrix(self, size, start, end):
        """Create an identity matrix with modified diagonal values."""
        matrix = np.identity(size)
        diagonal_values = np.linspace(start, end, num=size)
        np.fill_diagonal(matrix, diagonal_values)
        return matrix

    def _expand_matrix(self, matrix, new_rows, axis=0):
        """
        Expand the matrix with new rows or columns filled with zeros.
        """
        additional_shape = (
            (new_rows, matrix.shape[1]) if axis == 0 else (matrix.shape[0], new_rows)
        )
        return np.concatenate([matrix, np.zeros(additional_shape)], axis=axis)

    def _modify_matrix(self, matrix, coef_start, coef_end, size):
        """Modify part of the matrix diagonal with linearly spaced values."""
        np.fill_diagonal(
            matrix[:size, :size], np.linspace(coef_start, coef_end, num=size)
        )
        return matrix

    def preprocess_coef(self):
        """Preprocesses and returns the coefficient matrix based on the pre-configured parameters."""
        fr_coef = self._create_modified_identity_matrix(
            self.horizon_fr, self.theta_fr, self.theta_fr / 2
        )
        fr_coef = self._expand_matrix(
            fr_coef, self.horizon_sg - self.horizon_fr, axis=0
        )

        sg_coef = np.identity(self.horizon_sg) * self.theta_sg
        sg_coef = self._modify_matrix(
            sg_coef, self.theta_sg / 2, self.theta_sg, self.horizon_fr
        )

        return np.concatenate([fr_coef, sg_coef], axis=1)

    def preprocess_data(self, df):
        """Renames create_lagged_features to preprocess_data."""
        lagged_df = pd.DataFrame(index=df.index)
        # create lagged features for 'sg'
        for i in range(self.horizon_sg):
            lagged_df[f"sg_lag_{i+1}"] = df["sg"].shift(i)

        # create lagged features for 'fr'
        for i in range(self.horizon_fr):
            lagged_df[f"fr_lag_{i+1}"] = df["fr"].shift(i)

        # Drop initial rows where any lagged value would be NaN due to the shifting
        return lagged_df.dropna()

    def predict(self, df):
        """Renames forecast to predict."""
        coef_matrix = self.preprocess_coef()
        # Create and process lagged features
        lagged_df = self.preprocess_data(df)

        # Perform matrix multiplication for the forecast
        forecast_results = np.matmul(lagged_df.fillna(0).to_numpy(), coef_matrix.T)

        # Converting results to DataFrame
        return pd.DataFrame(
            forecast_results,
            index=lagged_df.index,
            columns=[f"Forecast_{i+1}" for i in range(coef_matrix.shape[0])],
        )


def main():
    # Sample usage
    np.random.seed(0)  # For reproducible results
    data = {
        "sg": np.random.rand(100),
        "fr": np.random.rand(100),
    }
    dates = pd.date_range(start="2022-01-01", periods=100)
    df = pd.DataFrame(data, index=dates)

    model = HexaLinearRegression(
        horizon_sg=14, horizon_fr=5, theta_sg=0.012, theta_fr=0.014
    )
    forecast_df = model.predict(df)
    print(forecast_df.tail())


if __name__ == "__main__":
    main()
