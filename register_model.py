import mlflow
import mlflow.pyfunc
from main import (
    ManualLR,
)  # Assuming ManualLR class is defined in main.py and adapted for MLflow as previously outlined
import pandas as pd
import numpy as np

# Change these variables as needed
experiment_name = "ManualLR_Experiment"
model_name = "ManualLR_Model"
artifact_path = "manual_lr_model"


def register_model(params):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log parameters (optional)
        mlflow.log_params({f"{k}_lag": v.get("lag") for k, v in params.items()})
        mlflow.log_params({f"{k}_theta": v.get("theta") for k, v in params.items()})
        mlflow.log_params(
            {f"{k}_confidence": v.get("confidence") for k, v in params.items()}
        )

        # Initialize your model with parameters
        model = ManualLR()
        model.fit(params=params)

        # Log model
        mlflow.pyfunc.log_model(artifact_path=artifact_path, python_model=model)

        # Register model in MLflow model registry
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"Model Version: {mv.version}")
        print("Registered model:", model_uri)


if __name__ == "__main__":
    # Define your parameters
    params = {
        "fruit": {"lag": 5, "theta": 0.014},
        "green": {"lag": 14, "theta": 0.012},
        # "fruitset": {"lag": 21, "theta": 0.001},
    }

    register_model(params=params)
