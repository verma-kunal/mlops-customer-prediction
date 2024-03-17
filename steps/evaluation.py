import logging
from zenml import step
import pandas as pd
import mlflow
from zenml.client import Client

from model.evaluation import (
    MSE, 
    RMSE, 
    R2Score
)
from sklearn.base import RegressorMixin # because we have a regression model as well (LR)
from typing_extensions import Annotated
from typing import Tuple

# initiate the experiment tracker object
# experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False)
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
    Annotated[float, "r2_score"], 
    Annotated[float, "rmse"]
]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:

        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)
        
        return r2_score, rmse # more efficient
    except Exception as e:
        logging.error(e)
        raise e