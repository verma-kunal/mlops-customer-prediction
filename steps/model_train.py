import logging
import pandas as pd
import mlflow

from zenml import step
from zenml.client import Client
from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


# initiate the experiment tracker object
experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,# choosing the model from "config.py" file
    y_test: pd.Series,
    config: ModelNameConfig, 
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
        elif config.model_name == "xgbm":
            mlflow.sklearn.autolog() # automatically logs the models, scores etc.
            model = XGBoostModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        elif config.model_name == "random_forest":
            mlflow.sklearn.autolog() # automatically logs the models, scores etc.
            model = RandomForestModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        else:
            raise ValueError("Model name not supported!")
        
        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e