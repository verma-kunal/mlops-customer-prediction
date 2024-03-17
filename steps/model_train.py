import logging
import pandas as pd
import mlflow

from zenml import step
from zenml.client import Client
from model.model_dev import (
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
)
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


# initiate the experiment tracker object
# experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig, # choosing the model from "config.py" file
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None

        if config.model_name == "linear_regression":
            mlflow.sklearn.autolog() # automatically logs the models, scores etc.
            model = LinearRegressionModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        elif config.model_name == "lightgbm":
            mlflow.sklearn.autolog() # automatically logs the models, scores etc.
            model = LightGBMModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        elif config.model_name == "random_forest":
            mlflow.sklearn.autolog() # automatically logs the models, scores etc.
            model = RandomForestModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        else:
            raise ValueError("Model name not supported!")
    except Exception as e:
        logging.error(e)
        raise e