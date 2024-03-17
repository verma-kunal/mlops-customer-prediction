import logging
from typing import Tuple
import pandas as pd

# importint the data cleaning model classes:
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing_extensions import Annotated
from zenml import step

# step for cleaning the data:
@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: raw data - pd.DataFrame
    Returns:
        x_train: training data
        x_test: testing data
        y_train: training labels
        y_test: testing labels
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()

        logging.info("Data Cleaning completed!")
        return x_train, x_test, y_train, y_test
    except Exception as err:
        logging.error(err)
        raise err