from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    # which model we want to use!    
    model_name: str = "random_forest" 
    fine_tuning: bool = False

    

    