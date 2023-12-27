import logging
import mlflow
from zenml import step
import pandas as pd

from sklearn.base import RegressorMixin

from mlops_maker.src.model_dev import LinearRegressionModel
from . import config
from .config import ModelNameConfig

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
) -> RegressorMixin:
    try:
        model = None
        if ModelNameConfig().model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.ModelNameConfig.model_name} not supported")
    except Exception as e:
        logging.error(f"Error in training model:{e}")
        raise e
