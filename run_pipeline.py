from pipelines.training_pipeline import train_pipeline
from zenml.client import Client
from steps.config import ModelNameConfig

# instantiate the class
model_config = ModelNameConfig()

if __name__ == "__main__":
    print("Using the model: ", model_config.model_name)
    print("MLFlow Check URI: ",Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/Users/kunalverma/Downloads/repos/customer-prediction-mlops/data/olist_customers_dataset.csv")