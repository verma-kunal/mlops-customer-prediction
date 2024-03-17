from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.model_train import train_model

@pipeline(enable_cache=False)
def train_pipeline(data_path: str):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        r2_score
        rmse
    """
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model,x_test,y_test)



    