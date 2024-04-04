import mlflow.pyfunc
import numpy as np
import pandas as pd


def load_model(model_name: str, model_version: str = None):
    """
    Loads a model from the MLflow model registry.

    Parameters:
    - model_name: The name of the model in the model registry.
    - model_version: The version of the model to load. If None, the latest version is loaded.

    Returns:
    - The loaded model.
    """
    if model_version is None:
        model_uri = f"models:/{model_name}/latest"
    else:
        model_uri = f"models:/{model_name}/{model_version}"

    # Load model as a PyFuncModel
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    return loaded_model


if __name__ == "__main__":
    # Define the model name and optionally version
    model_name = "ManualLR_Model"  # Use your model name
    model_version = None  # Specify if you want a specific version, e.g., "1"

    # Load the model
    model = load_model(model_name, model_version)

    np.random.seed(0)  # For reproducibility
    sample_data = {
        "fruit": np.random.rand(100),
        "green": np.random.rand(100),
        "fruitset": np.random.rand(100),
    }
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame(sample_data, index=dates)
    result = model.predict(df)
    print(result)

    print(
        f"Model loaded successfully: {model_name} (Version: {model_version if model_version else 'Latest'})"
    )
